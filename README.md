# statesman

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
    @statesman.event("Start a Process", None, States.starting)
    async def start(self, command: str) -> None:
        self.command = command
        self.pid = 31337
        self.logs.clear() # Flush logs between runs

    @statesman.event("Mark as process as running", source=States.starting, target=States.running)
    async def run(self, transition: statesman.Transition) -> None:
        self.logs.append(f"Process pid {self.pid} is now running (command=\"{self.command}\")")

    @statesman.event("Stop a running process", source=States.running, target=States.stopping)
    async def stop(self) -> None:
        self.logs.append(f"Shutting down pid {self.pid} (command=\"{self.command}\")")

    @statesman.event("Terminate a running process", source=States.stopping, target=States.stopped)
    async def terminate(self) -> None:
        self.logs.append(f"Terminated pid {self.pid} (\"{self.command}\")")
        self.command = None
        self.pid = None

    async def after_transition(self, transition: statesman.Transition) -> None:
        if transition.event and transition.event.name == "stop":
            await self.terminate()
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

## Quick start

Forthcoming after I get some sleep.

## License

statesman is licensed under the terms of the Apache 2.0 license.
