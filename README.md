# statesman

![Run Tests](https://github.com/opsani/statesman/workflows/Run%20Tests/badge.svg)
[![license](https://img.shields.io/github/license/opsani/statesman.svg)](https://github.com/opsani/statesman/blob/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/statesman.svg)](https://pypi.org/project/statesman/)
[![release](https://img.shields.io/github/release/opsani/statesman.svg)](https://github.com/opsani/statesman/releases/latest)
[![GitHub release
date](https://img.shields.io/github/release-date/opsani/statesman.svg)](https://github.com/opsani/statesman/releases)

![Statesman Logo](./docs/statesman_logo.png)

**The diplomatic path to building state machines in modern Python.**

statesman is a library that provides an elegant and expressive API for
implementing state machines in asynchronous Python 3.8+. It will negotiate
with complexity on your behalf and broker a clear, concise agreement about
how state is to be managed going forward.

## Features

* A lightweight, but fully featured implementation of the usual suspects in
  finite state machine libraries (states, events, transitions, actions).
* A declarative, simple API utilizing type hints, decorators, and enums.
* Provides a rich set of actions and callbacks for states and events. States
  support entry and exit actions, events have guard, before, on, and after
  actions. State machine wide callbacks are provided via overridable methods.
* Designed and built async native. Callbacks are dispatched asynchronously,
  making it easy to integrate with long running, event driven processes.
* Guard actions can cancel transition before any state changes are applied.
* Data can be modeled directly on the state machine subclass compliments of
  [Pydantic](https://pydantic-docs.helpmanual.io/).
* Events support the use of arbitrary associated parameter data that is made
  available to all actions. Parameters are matched against the signature of the
  receiving callable, enabling compartmentalization of concerns.
* Solid test coverage and documentation.

## Example

So... what's it look like? Glad you asked.

```python
from typing import Optional, List
import statesman


class ProcessLifecycle(statesman.StateMachine):
    class States(statesman.StateEnum):
        starting = "Starting..."
        running = "Running..."
        stopping = "Stopping..."
        stopped = "Terminated."

    # Track state about the process we are running
    command: Optional[str] = None
    pid: Optional[int] = None
    logs: List[str] = []

    # initial state entry point
    @statesman.event(None, States.starting)
    async def start(self, command: str) -> None:
        """"Start a process."""
        self.command = command
        self.pid = 31337
        self.logs.clear()  # Flush logs between runs

    @statesman.event(source=States.starting, target=States.running)
    async def run(self, transition: statesman.Transition) -> None:
        """Mark the process as running."""
        self.logs.append(f"Process pid {self.pid} is now running (command=\"{self.command}\")")

    @statesman.event(source=States.running, target=States.stopping)
    async def stop(self) -> None:
        """Stop a running process."""
        self.logs.append(f"Shutting down pid {self.pid} (command=\"{self.command}\")")

    @statesman.event(source=States.stopping, target=States.stopped)
    async def terminate(self) -> None:
        """Terminate a running process."""
        self.logs.append(f"Terminated pid {self.pid} (\"{self.command}\")")
        self.command = None
        self.pid = None

    @statesman.enter_state(States.stopping)
    async def _print_status(self) -> None:
        print("Entering stopped status!")

    @statesman.after_event("run")
    async def _after_run(self) -> None:
        print("running...")

    async def after_transition(self, transition: statesman.Transition) -> None:
        if transition.event and transition.event.name == "stop":
            await self.terminate()


async def _examples():
    # Let's play.
    state_machine = ProcessLifecycle()
    await state_machine.start("ls -al")
    assert state_machine.command == "ls -al"
    assert state_machine.pid == 31337
    assert state_machine.state == ProcessLifecycle.States.starting

    await state_machine.run()
    assert state_machine.logs == ['Process pid 31337 is now running (command="ls -al")']

    await state_machine.stop()
    assert state_machine.logs == [
        'Process pid 31337 is now running (command="ls -al")',
        'Shutting down pid 31337 (command="ls -al")',
        'Terminated pid 31337 ("ls -al")',
    ]

    # Or start in a specific state
    state_machine = ProcessLifecycle(state=ProcessLifecycle.States.running)

    # Transition to a specific state
    await state_machine.enter_state(ProcessLifecycle.States.stopping)

    # Trigger an event
    await state_machine.trigger_event("stop", key="value")
```

States are defined as Python enum classes. The name of the enum item defines a
symbolic name for the state and the value provides a human readable description.
A class named `States` embedded within a state machine subclass is automatically
bound to the state machine.

Events are declared using the event decorator and the define the source and
target states of a transition. A source state of `None` defines an initial state
transition.

Once a method is decorated as an event action, the original method body is
attached to the new event as an on event action and the method is replaced with
a implementation that triggers the newly created event.

Actions can be attached to events at declaration time or later on via the
`guard_event`, `before_event`, `on_event`, and `after_event` decorators. Actions
can likewise be attached to states via the `enter_state` and `exit_state`
decorators.

There is an extensive API for working with the state machine and its components
programmatically.

## Why statesman?

Statesman was developed because we couldn't find anything like it. While there
is an embarassment of riches in the Python community with regard to FSM
libraries, many have legacy ties back to the Python 2 era and address our first
class requirements as add-ons rather than core functionality.

Our design goals were roughly:

* Utilize type hints extensively as both a documentation and design by contract
    tool to provide an expressive, readable API.
* Embrace asyncio as a first class citizen.
* Implement state machines as plain old Python objects with an easily understood
    method and input/output external API surface. Callers shouldn't need to know
    or care about FSM minutiae.
* Deliver an API closely aligned with the [UML State Machine](https://en.wikipedia.org/wiki/UML_state_machine)
conceptual framework that most developers will have exposure to.
* Shun famous string values in favor of Python [`enum`](https://docs.python.org/3/library/enum.html)
subclasses to facilitate type enforcement, refactoring, and IDE completions.
* Enable state machines to be modeled programmatically or declaratively.
* Provide robust support for modeling data within the state machine class itself
    and passing external data into transitions.
* Disallow implicit "magical" behaviors such as automatically creating states,
    events, and actions objects or dynamically defining/dispatching methods.

Ultimately, we really wanted something that would fit right in with our existing
coding style and API aesthetics. Maybe statesman matches yours too.

## Transition Lifecycle

Statesman executes actions during a transition in a defined order. The
`statesman.Transition` class is a callable that is responsible for modeling
a state change and coordinating its execution. The table below describes the
order of operations performed when a transition is called.

| Target | Current State | Comments |
|--------|---------------|----------|
| `StateMachine.guard_transition` | `source` | Method dispatch for subclasses. Can cancel the transition. |
| `StateMachine.before_transition` | `source` | Method dispatch for subclasses. |
| `Event.actions.guard` | `source` | Can cancel the transition. |
| `Event.actions.before` | `source` | |
| `State.actions.exit` | `source` | |
| `StateMachine.on_transition` | `target` | Method dispatch for subclasses. |
| `Event.actions.on` | `target` | |
| `State.actions.entry` | `target` | |
| `Event.actions.after` | `target` | |
| `StateMachine.after_transition` | `target` | Method dispatch for subclasses. |

## API Overview

Statesman is extensively covered with docstrings and automated tests. The
proceding subsections present a non-exhaustive task oriented overview of the
API. Check the docstrings and look through the tests if you don't find exactly
what you are looking for.

### Initial States

Statesman considers a transition in which the source state is `None` to describe
an initial state transition. There are a couple of ways to describe initial
states within statesman:

```python
import statesman


# Describe the initial state with `statesman.InitialState`
class StateMachine(statesman.StateMachine):
    class States(statesman.StateEnum):
        starting = 'Starting...'
        running = 'Running...'
        stopping = 'Stopping...'
        stopped = statesman.InitialState('Terminated.')

    @statesman.event(None, States.starting)
    def start(self) -> None:
        ...


async def _example() -> None:
    # Set at initialization time
    state_machine = StateMachine(state=StateMachine.States.stopping)

    # Enter a state directly
    state_machine = StateMachine()
    await state_machine.enter_state(StateMachine.States.running)

    # Via an event
    state_machine = StateMachine()
    await state_machine.start()
```

### Introspecting State

Each state machine instance has a `state` attribute that is the source of truth
for the state machine. The state can be compared to `statesman.State` object
instances or string values.

```python
import statesman


class StateMachine(statesman.StateMachine):
    class States(statesman.StateEnum):
        starting = 'Starting...'
        running = 'Running...'
        stopping = 'Stopping...'
        stopped = 'Terminated.'


async def _example() -> None:
    state_machine = StateMachine(state=StateMachine.States.stopping)
    state_machine.state == StateMachine.States.stopping  # => True
    state_machine.state == "stopping"  # => True
    state_machine.state == StateMachine.States.running  # => False
    state_machine.state == "stopped"  # => False
```

### Entering States

States can be directly entered via the `statesman.StateMachine.enter_state`
method. States can be referenced by name or by `stateman.State` object instance.
When a state is entered directly, a transition is triggered between the source
and target states:

```python
import statesman


class StateMachine(statesman.HistoryMixin, statesman.StateMachine):
    class States(statesman.StateEnum):
        first = "1"
        second = "2"
        third = "3"


async def _example() -> None:
    state_machine = StateMachine()
    await state_machine.enter_state(StateMachine.States.first)
    await state_machine.enter_state(StateMachine.States.second)
    await state_machine.enter_state(StateMachine.States.third)
```

The type of transition performed is configurable (see below).

Note that `enter_state` should be used thoughtfully as it enables transitions
that may not be expressible via events. Its behavior can also be constrained
and customized.

### Transition Types

There are three types of transitions that can be performed by statesman. The
most common type is an external transition in which the machine moves between
two distinct states. When a transition occurs in which the source and target
states are the same, there two other possible modes: `internal` and `self`.

* `statesman.Transition.Types.external`: A transition in which the state is
    changed from one value to another.
* `statesman.Transition.Types.internal`: A transition in which the source and
    target states are the same but are not exited and reentered during the transition.
* `statesman.Transition.Types.self`: A transition in which the source and
    target states are the same and are exited and reentered during the transition.

### Transition Return Values

To enable external consumers to interact with the state machine without being
exposed to its implementation details, Statesman supports a flexible set of
return values from transitions.

A default return type can be configured when defining an event via the
decorators or programmatic interface. An explicit return value type can also be
requested when a transition is dispatched through the `statesman.StateMachine.enter_state`
or `statesman.StateMachine.trigger_event` methods.

Transitions can invoke an arbitrary number of actions and the desired return
value semantics vary from case to case. Think about the API you wish to present
to the developer and carefully consider if/how `None` and `False` values are
utilized within the state machine.

The available return types are:

* `bool`: A boolean value that indicates if the transition completed
    successfully.
* `object`: An arbitrary output value returned by the transition.
* `tuple`:  A tuple value containing a boolean and output object value.
* `list`: A list of results returned by all actions invoked by the transition.
* `statesman.Transition`: The transition object itself, containing all details
    about the transition.

Note that return types are configured as **type** arguments:

```python
import statesman


class StateMachine(statesman.StateMachine):
    class States(statesman.StateEnum):
        starting = 'Starting'
        running = 'Running'
        stopping = 'Stopping'
        stopped = 'Stopped'

    @statesman.event(None, States.starting, return_type=bool)
    async def start(self) -> int:
        return 31337


async def _example() -> None:
    state_machine = await StateMachine.create()
    bool_result = await state_machine.trigger_event("start")
    int_result = await state_machine.start(return_type=object)
    transition = await state_machine.enter_state(
        StateMachine.States.stopped,
        return_type=statesman.Transition
    )
    print(
        f"Return Values: state_machine={state_machine}\n"
        f"bool_result={bool_result}\n"
        f"int_result={int_result}\n"
        f"transition={transition}"
    )
```

### Defining and Triggering Events

Events are typically defined using the `statesman.event` decorator.
Each event has a **source** state and a **target** state. Typically these are
distinct states and describe an event that triggers an **external** transition.
When the source and target state are the same, the event describes an
**internal** or **self** transition (see above for details).

The source and target states can be described by string name, enum member
value, or the special sentinel values of `statesman.StateEnum.__any__` or
`statesman.StateEnum.__active__`. The `__any__` sentinel describes a source
state of any member of the state enumeration (but not `None`). The `__active__`
sentinel resolves dynamically to the currently active state of the state
machine and is useful in cases where you want to define a reflexive internal or
self transition that can be triggered from several states without duplicating
logic.

Events can be triggered programmatically by name, method, or by
calling a decorated event method.

```python
import statesman


class StateMachine(statesman.StateMachine):
    class States(statesman.StateEnum):
        waiting = 'Waiting'
        running = 'Running'
        stopped = 'Stopped'
        aborted = 'Aborted'

    @statesman.event(None, States.waiting)
    async def start(self) -> None:
        ...

    @statesman.event(States.waiting, States.running)
    async def run(self) -> None:
        ...

    @statesman.event(States.running, States.stopped)
    async def stop(self) -> None:
        ...

    @statesman.event(States.__any__, States.aborted)
    async def abort(self) -> None:
        ...

    @statesman.event(
        States.__any__,
        States.__active__,
        type=statesman.Transition.Types.self
    )
    async def check(self) -> None:
        print("Exiting and reentering active state!")


async def _example() -> None:
    state_machine = await StateMachine.create()
    await state_machine.trigger_event("start")
    await state_machine.run()
    await state_machine.trigger_event(state_machine.stop)
```

### State and Event Actions

States and Events support the attachment of an arbitrary number of actions. An
action is an object that wraps a callable that called at a designated moment in
the lifecycle of the state machine.

State objects support the following action types:

* `statesman.Action.Types.entry`: Called when a state is entered during a
    transition.
* `statesman.Action.Types.exit`: Called when a state is exited during a
    transition.

Event actions support the following action types:

* `statesman.Action.Types.guard`: Called to determine if the event can be
    executed.
* `statesman.Action.Types.before`: Called before the state transition described
    by the event is applied. The state of the machine is the source state.
* `statesman.Action.Types.on`: Called when the state transition described by the
    event is applied. The state of the machine is the target state.
* `statesman.Action.Types.after`: Called after the state transition described by
    the event has been applied. The state of the machine is the target state.

State and Event actions can be defined in several ways. The `statesman.state`
and `statesman.event` decorators accept keyword arguments named after the action
types that they support. These arguments support method object references,
names, or callables (e.g., lambdas). Additionally, there are standalone
decorators that enable a declarative style of action definition.

```python
import random
import statesman


class StateMachine(statesman.StateMachine):
    class States(statesman.StateEnum):
        starting = 'Starting...'
        running = 'Running...'
        stopping = 'Stopping...'
        stopped = statesman.InitialState('Terminated.')

    @statesman.event(States.stopped, States.starting)
    async def start(self) -> None:
        ...

    @statesman.enter_state(States.starting)
    async def _announce_start(self) -> None:
        print("enter:starting")

    def _can_run(self) -> bool:
        return False

    @statesman.event(States.starting, States.running, guard=_can_run)
    async def run(self) -> None:
        ...

    @statesman.event(States.running, States.stopping)
    async def stop(self) -> None:
        ...

    @statesman.after_event(stop)
    async def _announce_stop(self) -> None:
        print("after:stop")

    @statesman.event(
        States.stopping,
        States.stopped,
        guard=lambda: random.choice([True, False])
    )
    async def terminate(self) -> None:
        ...
```

### Guard Callbacks and Actions

Guard callbacks and actions are handled differently from other behavioral hooks.
Because guards can be used to cancel/reject an event, they are executed
sequentially in the order that they were added to the state machine.

Upon encountering a failing guard that has returned `False` or raised an
exception of type `AssertionError`, the state machine consults the `guard_with`
behavior configured on the `Config` class nested within the state machine class.

There are three behaviors available for configuration via `guard_with`:

* `statesman.Guard.silence` (Default): The transition is aborted and an empty
    result set is returned.
* `statesman.Guard.warning`: The transition is aborted, an empty result set is
    returned, and a warning is logged.
* `statesman.Guard.exception`: The transition is aborted and a `RuntimeError` is
    raised.

```python
import statesman


class StateMachine(statesman.StateMachine):
    class States(statesman.StateEnum):
        starting = 'Starting...'
        running = 'Running...'
        stopping = 'Stopping...'
        stopped = statesman.InitialState('Terminated.')

    class Config:
        guard_with = statesman.Guard.exception
```

### Passing Data to Transitions

Statesman is designed to model and manage arbitrary data that is bound to the
current state of the state machine. Such data can be provided as positional and
keyword arguments when a transition is triggered via the `statesman.StateMachine.enter_state`
or `statesman.StateMachine.trigger_event` methods.

Statesman utilizes method argument list introspection and type-hinting to invoke
callbacks and actions with the specific arguments that they are interested in.
This facilitates cleaner encapsulation, separation of concerns, and
maintainability by ensuring that each callback and method declares its
expectations. The `*args` and `**kwargs` variadic arguments can be used as
necessary.

```python
import statesman


class StateMachine(statesman.StateMachine):
    class States(statesman.StateEnum):
        starting = 'Starting...'
        running = 'Running...'
        stopping = 'Stopping...'
        stopped = statesman.InitialState('Terminated.')

    @statesman.event(States.stopped, States.starting)
    async def start(self, process_name: str) -> None:
        ...

    @statesman.event(States.starting, States.running)
    async def run(self, uid: int, gid: int) -> None:
        ...

    @statesman.event(States.running, States.stopping)
    async def stop(self, *args, all: bool = False, **kwargs) -> None:
        ...


async def _example() -> None:
    state_machine = await StateMachine.create()
    await state_machine.start("servox")
    await state_machine.run(0, 31337)
    await state_machine.stop("one", "two", all=True, this="That")
```

### State Machine Inheritable Callbacks

The `statesman.StateMachine` class is typically subclassed. The base class
provides inheritable method implementations of transition lifecycle events.
Analogous functionality can be implemented through callbacks as through the
State and Event actions except that perspective is rooted on the transition
as opposed to the State or Event.

See the [Transition Lifecycle](#transition-lifecycle) for specifics on execution
order and precedence between callbacks and actions.

```python
import statesman


class InheritableStateMachine(statesman.StateMachine):
    async def guard_transition(self, transition: statesman.Transition, *args, **kwargs) -> bool:
        return True

    async def before_transition(self, transition: statesman.Transition, *args, **kwargs) -> None:
        ...

    async def on_transition(self, transition: statesman.Transition, *args, **kwargs) -> None:
        ...

    async def after_transition(self, transition: statesman.Transition, *args, **kwargs) -> None:
        ...
```

### Async Initialization

State machines can be asynchronously constructed and initialized:

```python
import statesman


class States(statesman.StateEnum):
    starting = 'Starting'
    running = 'Running'
    stopping = 'Stopping'
    stopped = 'Stopped'


async def _example() -> None:
    state_machine = await statesman.StateMachine.create(
        states=statesman.State.from_enum(States)
    )
    print(f"Initialized state machine: {repr(state_machine)}")
```

### Restricting State Entry

The `statesman.StateMachine.enter_state` method provides great flexibility for
putting the state machine into a specific state without regard for the event
transition constraints that have been defined. Depending on the specifics of the
state being modeled, this can become a potential liability as it can enable
programmer errors by coercing an initialized state machine into an inconsistent
or unexpected state. It can be desirable to restrict the use of `enter_state` to
establishing an initial state or forbidding its use entirely in favor of initial
state transition events.

As such, statesman provides a set of behaviors that govern how the `enter_state`
method behaves. The behavior is configured via the `state_entry` attribute of
the `Config` class nested within the state machine class.

There are four behaviors available for configuration via `state_entry`:

* `statesman.Entry.allow` (Default): The `enter_state` method can be called at
    any time.
* `statesman.Entry.initial`: The `enter_state` method can be called to
    establish an initial state and thereafter is forbidden.
* `statesman.Entry.ignore`: The `enter_state` method can never succeed and
    will fail and return silently when called.
* `statesman.Entry.forbid`: The `enter_state` method can never succeed and will
    raise an exception when called.

```python
import statesman


class StateMachine(statesman.StateMachine):
    class States(statesman.StateEnum):
        starting = 'Starting...'
        running = 'Running...'
        stopping = 'Stopping...'
        stopped = statesman.InitialState('Terminated.')

    class Config:
        state_entry = statesman.Entry.forbid
```

### Tracking History

It can be interesting to maintain a history of transitions that have occurred
within a state machine. Statesman provides out of the box support for history
tracking via the `statesman.HistoryMixin` class:

```python
import statesman


class StateMachine(statesman.HistoryMixin, statesman.StateMachine):
    class States(statesman.StateEnum):
        ready = statesman.InitialState("Ready")
        analyzing = "Analyzing"

        awaiting_description = "Awaiting Description"
        awaiting_measurement = "Awaiting Measurement"
        awaiting_adjustment = "Awaiting Adjustment"

        done = "Done"
        failed = "Failed"


async def _example() -> None:
    state_machine = StateMachine()
    await state_machine.enter_state(StateMachine.States.analyzing)
    await state_machine.enter_state(StateMachine.States.awaiting_measurement)
    await state_machine.enter_state(StateMachine.States.done)
    await state_machine.enter_state(StateMachine.States.failed)

    print(f"The history is: {state_machine.history}")
```

### Sequencing Transitions

There are use cases such as testing and protocol development in which it can
become desirable for the state machine to follow a defined linear sequence of
transitions.

Statesman supports such use cases via the `statesman.SequencingMixin` class.
Transitions can be sequenced by passing **coroutine** objects into the
`sequence` method. Coroutine objects are returned when you call an async
function but do not await its execution. The coroutine objects freeze your
desired state transition, which are queued for iterative execution by invoking
the `next_transition` method.

```python
from typing import List
import statesman


class StateMachine(statesman.SequencingMixin, statesman.StateMachine):
    class States(statesman.StateEnum):
        ready = statesman.InitialState("Ready")
        analyzing = "Analyzing"

        awaiting_description = "Awaiting Description"
        awaiting_measurement = "Awaiting Measurement"
        awaiting_adjustment = "Awaiting Adjustment"

        done = "Done"
        failed = "Failed"

    @statesman.event([States.ready, States.analyzing], States.awaiting_description)
    async def request_description(self) -> None:
        """Request a Description of application state from the servo."""
        ...

    @statesman.event([States.ready, States.analyzing], States.awaiting_measurement)
    async def request_measurement(self, metrics: List[str]) -> None:
        """Request a Measurement from the servo."""
        ...

    @statesman.event([States.ready, States.analyzing], States.awaiting_adjustment)
    async def recommend_adjustments(self, adjustments: List[str]) -> None:
        """Recommend Adjustments to the Servo."""
        ...


async def _example() -> None:
    state_machine = StateMachine()
    state_machine.sequence(
        state_machine.request_description(),
        state_machine.request_measurement(metrics=[...]),
        state_machine.recommend_adjustments([...]),
        state_machine.request_measurement(metrics=[...]),
        state_machine.recommend_adjustments([...]),
    )

    while True:
        transition = state_machine.next_transition()
        print(f"Executed transition: {repr(transition)}")
        if not transition:
            break
```

## Future Directions

There is active work underway exploring alternate syntaxes for working with
statesman via a state table syntax/interface.

It would also be interesting to follow in the tradition of other FSM libraries
that have come before and support graphing/visualizations of statesman machines.

It seems inevitable that hierarchical state machine support will eventually
land.

## Acknowledgements

[Klemen Verdnik](https://github.com/chipxsd) contributed the statesman logo.

## License

statesman is licensed under the terms of the Apache 2.0 license.
