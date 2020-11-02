"""Statesman is a modern state machine library."""
import asyncio
import collections
import contextlib
import datetime
import enum
import functools
import inspect
import types
import typing
from inspect import isclass
from typing import AsyncIterator, Any, ClassVar, Dict, List, Literal, Mapping, Optional, Callable, Sequence, Type, Tuple, Union

import pydantic


__all__ = [
    "States",
    "State",
    "Transition",
    "Event",
    "StateMachine",
    "event",
    "on_state",
    "enter_state",
    "exit_state"
]

class StateEnum(enum.Enum):
    """An abstract enumeration base class for defining states within a state machine.
    
    State enumerations are interpreted as describing states where the `name` attribute defines the unique, 
    symbolic name of the state within the state machine while the `value` attribute defines the human readable 
    description.
    """
    pass

class Action(pydantic.BaseModel):
    """An Action is a callable object attached to states and events within a state machine."""
    class Types(str, enum.Enum):
        """An enumeration that defines the types of actions that can be attached to states and events."""
                
        # State actions
        entry = "entry"
        exit = "exit"
        
        # Event actions
        guard = "guard"
        before = "before"
        on = "on"
        after = "after"
    
    callable: Callable
    signature: inspect.Signature
    type: Optional[Types] = None
    
    @pydantic.root_validator(pre=True)
    @classmethod
    def _cache_signature(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if callable := values.get("callable", None):
            values["signature"] = inspect.Signature.from_callable(callable)
        
        return values
    
    async def __call__(self, *args, **kwargs) -> Any:
        """Call the action with the matching parameters and return the result."""
        matched_args, matched_kwargs = _parameters_matching_signature(self.signature, *args, **kwargs)
        if asyncio.iscoroutinefunction(self.callable):
            return await self.callable(*matched_args, **matched_kwargs)
        else:
            return self.callable(*matched_args, **matched_kwargs)
    
    class Config:
        arbitrary_types_allowed = True


class BaseModel(pydantic.BaseModel):
    """Provides common functionality for statesman models."""
    _actions: List[Action] = pydantic.PrivateAttr([])
        
    def _add_action(self, action: Action) -> None:
        """Add a action."""
        self._actions.append(action)
    
    def _remove_action(self, action: Action) -> None:
        """Remove a action."""
        self._actions.remove(action)
    
    def _remove_actions(self, actions: Union[None, List[Action], Action.Types] = None) -> None:
        """Remove a collection of actions."""
        if actions is None:
            self._actions = []
        elif isinstance(actions, list):
            for action in self._actions:
                if action in actions:
                    self._actions.remove(action)
        elif isinstance(actions, Action.Types):
            for action in self._actions:
                if action.type == actions:
                    self._actions.remove(action)
        else:
            raise ValueError(f"invalid argument: {actions}")
    
    def _get_actions(self, type_: Action.Types) -> List[Action]:
        """Retrieve a subset of actions by type."""
        return list(filter(lambda c: c.type == type_, self._actions))

    async def _run_actions(self, type_: Action.Types, *args, **kwargs) -> List[Any]:
        return await asyncio.gather(*(action(*args, **kwargs) for action in self._get_actions(type_)))


class State(BaseModel):
    """Models a state within a state machine.
    
    State objects can be tested for equality against `str` and `StateEnum` objects.
    """
    name: str
    description: Optional[str] = None
    
    @classmethod
    def from_enum(cls, class_: Type[StateEnum]) -> List['State']:
        """Return a list of State objects from a state enum subclass."""
        states = []
        if isclass(class_) and issubclass(class_, StateEnum):
            for item in class_:
                states.append(cls(name=item.name, description=item.value))
        else:
            raise TypeError(f"invalid parameter: \"{class_.__class__.__name__}\" is not a StateEnum subclass: {class_}")
        
        return states

    @pydantic.validator("name", "description", pre=True)
    @classmethod
    def _value_from_base_states(cls, value: Union[str, StateEnum], field) -> str:
        """Extract the appropriate value for the model field from a States enumeration value.
        
        States objects are serialized differently than typical Enum values in Pydantic. The name field
        is used to populate the state name and the value populates the description.
        """
        if isinstance(value, StateEnum):
            if field.name == "name":
                return value.name
            elif field.name == "description":
                return value.value
                
        return value
    
    def __init__(self, name: str, description: Optional[str] = None) -> None:
        super().__init__(name=name, description=description)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, StateEnum):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            return super().__eq__(other)
    
    @property
    def actions(self) -> List[Action]:
        """Return a list of entry and exit actions attached to the state."""
        return super()._actions.copy()

    def get_actions(self, type_: Literal[Action.Types.entry, Action.Types.exit]) -> List[Action]:
        """Return a list of all entry or exit actions attached to the state."""
        return super()._get_actions(type_)
    
    def add_action(self, callable: Callable, type_: Literal[Action.Types.entry, Action.Types.exit]) -> Action:
        """Add an entry or exit action to the state."""
        acceptable_types = (Action.Types.entry, Action.Types.exit)
        if type_ not in acceptable_types:
            raise ValueError(f"cannot add state action with type \"{type_}\": must be {_summarize(acceptable_types, conjunction='or', quote=True)}")
        action = Action(callable=callable, type=type_)
        super()._add_action(action)
        return action

    def remove_action(self, action: Action) -> Action:
        """Remove a action from the state."""
        return super()._remove_action(action)

    def remove_actions(
        self, actions: Union[None, List[Action], Literal[Action.Types.entry, Action.Types.exit]] = None
    ) -> None:
        """Remove actions that are attached to the state.
        
        There are three modes of operation:
        - Passing a value of `None` (the default) will remove all actions.
        - Passing a specific set of `Action` objects will remove only those actions. 
        - Passing `Action.Types.enter` or `Action.Types.exit` will remove all actions that match the given type.
        """
        return super()._remove_actions(actions)

class Event(BaseModel):
    """Event objects model something that happens within a state machine that triggers a transition from one state to another.
    
    Attributes:
        name: A unique name of the event within the state machine.
        description: An optional description of the event.
        sources: A list of states that the event can be triggered from. The inclusion of `None` denotes an initialization event.
        target: The state that the state machine will transition into at the completion of the event.
    """
    name: str
    description: Optional[str] = None    
    sources: List[Union[None, State]]
    target: State
    
    @property
    def actions(self) -> List[Action]:
        """Return a list of actions attached to the event."""
        return self._actions.copy()

    def get_actions(
        self, type_: Literal[Action.Types.guard, Action.Types.before, Action.Types.after]
    ) -> List[Action]:
        """Return a list of all guard, before, or after actions attached to the event."""
        return super()._get_actions(type_)
    
    def add_action(
        self, 
        callable: Callable, 
        type_: Literal[Action.Types.guard, Action.Types.before, Action.Types.on, Action.Types.after]
    ) -> Action:
        """Add a guard, before, on, or after action to the event."""
        acceptable_types = (Action.Types.guard, Action.Types.before, Action.Types.on, Action.Types.after)
        if type_ not in acceptable_types:
            raise ValueError(f"cannot add state action with type \"{type_}\": must be {_summarize(acceptable_types, conjunction='or', quote=True)}")
        action = Action(callable=callable, type=type_)
        super()._add_action(action)
        return action

    def remove_action(self, action: Action) -> Action:
        """Remove a action from the state."""
        return super()._remove_action(action)

    def remove_actions(
        self, actions: Union[None, List[Action], Literal[Action.Types.entry, Action.Types.exit]] = None
    ) -> None:
        """Remove actions that are attached to the state.
        
        There are three modes of operation:
        - Passing a value of `None` (the default) will remove all actions.
        - Passing a specific set of `Action` objects will remove only those actions. 
        - Passing `Action.Types.enter` or `Action.Types.exit` will remove all actions that match the given type.
        """
        return super()._remove_actions(actions)

class StateMachine(pydantic.BaseModel):
    """StateMachine objects model state machines comprised of states, events, and associated actions.
    
    Args:
        states: A list of states to add to the state machine.
        events: A list of events to add to the state machine.
        state: The initial state of the state machine. When `None` the state machine initializes into an 
            indeterminate state. The `enter_state` and `trigger` methods can be used to establish an initial
            state post-initialization.
    # TODO: add a note about using enter_state if you need args and init not calling entry callbacks
    """
    # __state__: ClassVar[Optional[StateEnum]] = None
    __state__: Optional[StateEnum] = None
    
    _state: Optional[State] = pydantic.PrivateAttr(None)
    _states: List[State] = pydantic.PrivateAttr([])
    _events: List[Event] = pydantic.PrivateAttr([])
    
    def __init__(self, states: List[State] = [], events: List[Event] = [], state: Optional[Union[State, str, StateEnum]] = None) -> None:
        super().__init__()
        
        # Initialize private attributes
        if states:
            self._states.extend(states)
                    
        if events:
            self._events.extend(events)
            
        # Handle embedded States class
        state_enum = getattr(self.__class__, "States", None)
        if state_enum:
            if not issubclass(state_enum, StateEnum):
                raise TypeError("States class must be a subclass of StateEnum")
            self._states.extend(State.from_enum(state_enum))
        
        # Handle type hints from __state__
        if not state_enum:
            type_hints = typing.get_type_hints(self.__class__)
            state_hint = type_hints["__state__"]
            if isclass(state_hint) and issubclass(state_hint, StateEnum):
                self._states.extend(State.from_enum(state_hint))
            else:
                # Introspect the type hint
                type_origin = typing.get_origin(state_hint)
                if type_origin is typing.Union:
                    args = typing.get_args(state_hint)

                    for arg in args:
                        if isclass(arg) and issubclass(arg, StateEnum):
                            self._states.extend(State.from_enum(arg))
                else:
                    # TODO: Handle other reasonable hints
                    raise TypeError(f"unsupported type hint: \"{state_hint}\"")
        
        # Initial state
        if isinstance(state, State):
            if state not in self._states:
                raise ValueError(f"invalid initial state: the state object given is not in the state machine")
            self._state = state
        
        elif isinstance(state, (StateEnum, str)):            
            state_ = self.get_state(state)
            if not state_:
                raise LookupError(f"invalid initial state: no state was found with the name \"{state}\"")
            self._state = state_
        
        elif state is None:
            # Assign from __state__ attribute if defined
            if initial_state := getattr(self.__class__, "__state__", None):
                state_ = self.get_state(initial_state)
                if not state_:
                    raise LookupError(f"invalid initial state: no state was found with the name \"{initial_state}\"")
                self._state = state_
                
        else:
            raise TypeError(f"invalid initial state: unexpected value of type \"{state.__class__.__name__}\": {state}")
        
        # Initialize any decorated methods
        for name, method in self.__class__.__dict__.items():
            if descriptor := getattr(method, "__event_descriptor__", None):
                target = self.get_state(descriptor.target)
                if not target:
                    raise ValueError(f"event creation failed: target state \"{descriptor.target}\" is not in the state machine")
                
                source_names = list(filter(lambda s: s is not None, descriptor.source))
                sources = self.get_states(*source_names)
                if None in descriptor.source:
                    sources.append(None)

                event = Event(
                    name=method.__name__,
                    description=descriptor.description,
                    sources=sources,
                    target=target,
                )
                
                # Create bound methods and attach them as actions
                for type_ in Action.Types:
                    if not hasattr(descriptor, type_.name):
                        continue
                    
                    callables = getattr(descriptor, type_.name)
                    for callable in callables:
                        event.add_action(
                            types.MethodType(callable, self), 
                            type_
                        )
                        self.add_event(event)
            elif descriptor := getattr(method, "__action_descriptor__", None):
                if descriptor.model == State:
                    obj = self.get_state(descriptor.name)
                    if not obj:
                        raise LookupError(f"unknown state: \"{descriptor.name}\"")
                elif descriptor.model == Event:
                    obj = self.get_event(descriptor.name)
                    if not obj:
                        raise LookupError(f"unknown event: \"{descriptor.name}\"")
                else:
                    raise TypeError(f"unknown model type: {descriptor.model.__name__}")
                
                # Create a bound method and attach the action
                obj.add_action(
                    types.MethodType(descriptor.callable, self), 
                    descriptor.type
                )
    
    @property
    def state(self) -> Optional[State]:
        """Return the current state f the state machine."""
        return self._state
    
    @property
    def states(self) -> List[State]:
        """Return the list of states in the machine."""
        return self._states.copy()
    
    def add_state(self, state: State) -> None:
        """Add a state to the machine."""
        self._states.append(state)
    
    def add_states(self, states: List[State]) -> None:
        """Add a list of states to the machine."""
        self._states.extend(states)
    
    def remove_state(self, state: State) -> None:
        """Remove a state from the machine."""
        self._states.remove(state)
    
    def remove_states(self, states: List[State]) -> None:
        """Remove a list of states from the machine."""
        for state in states:
            self.remove(state)
    
    def get_state(self, name: Union[str, StateEnum]) -> Optional[State]:
        """Retrieve a list of states in the state machine by name."""
        name_ = name.name if isinstance(name, StateEnum) else name
        return next(filter(lambda s: s.name == name_, self.states), None)
    
    def get_states(self, *names: List[Union[str, StateEnum]]) -> List[State]:
        """Retrieve a list of states in the state machine by name."""
        names_ = []
        for name in names:
            if isclass(name) and issubclass(name, StateEnum):
                names_.extend(list(map(lambda i: i.name, name)))
            elif isinstance(name, (StateEnum, str)):
                name_ = name.name if isinstance(name, StateEnum) else name
                names_.append(name_)
            else:
                raise TypeError(f"cannot get state for type \"{name.__class__.__name__}\": {name}")
        
        return list(filter(lambda s: s.name in names_, self.states))
    
    @property
    def events(self) -> List[Event]:
        """Return the list of events in the state machine."""
        return self._events.copy()
    
    def add_event(self, event: Event) -> None:
        """Add an event to the state machine."""
        self._events.append(event)
    
    def add_events(self, states: List[State]) -> None:
        """Add a list of events to the machine."""
        self._events.extend(states)
    
    def remove_event(self, event: Event) -> None:
        """Remove an event from the state machine."""
        self._events.remove(event)
    
    def get_event(self, name: Union[str, StateEnum]) -> Optional[Event]:
        """Return the event with the given name or None if not found."""
        if isinstance(name, (str, StateEnum)):
            name_ = name.name if isinstance(name, StateEnum) else name
        else:
            raise TypeError(f"cannot get event for name of type \"{name.__class__.__name__}\": {name}")
        
        return next(filter(lambda e: e.name == name_, self._events), None)

    # TODO: cannot add an event to the machine that references an unknown state
    # TODO: when adding/removing a state or event, remove all the transitions it is referenced in
    # TODO: check that state and event aren't already in the state machine
    # _states = States.to_states()
    # _initial = States.starting
    # __state__ = States.starting
    # _state: States = States.starting # TODO: Find the field and update its signature
    
    async def trigger(self, event: Union[Event, str], *args, **kwargs) -> bool:
        """Trigger a state transition event.
        
        The state machine must be in a source state of the event being triggered. Initial event transitions
        can be triggered for events that have included `None` in their source states list.
        
        Args:
            event: The event to trigger a state transition with.
            args: Supplemental positional arguments to be passed to the transition and triggered actions.
            kwargs: Supplemental keyword arguments to be passed to the transition and triggered actions.
        
        Returns:
            A boolean value indicating if the transition was successful.
        
        Raises:
            ValueError: Raised if the event object is not a part of the state machine.
            LookupError: Raised if the event cannot be found by name.
            TypeError: Raised if the event value given is not an Event or str object.
        """        
        if isinstance(event, Event):
            event_ = event
            if event_ not in self._events:
                raise ValueError(f"event trigger failed: the event object given is not in the state machine")
            
        elif isinstance(event, str):
            event_ = self.get_event(event)
            if not event_:
                raise LookupError(f"event trigger failed: no event was found with the name \"{event}\"")
            
        else:
            raise TypeError(f"event trigger failed: cannot trigger an event of type \"{event.__class__.__name__}\": {event}")
        
        if self.state not in event_.sources:
            if self.state:
                raise RuntimeError(f"event trigger failed: the \"{event_.name}\" event cannot be triggered from the current state of \"{self.state.name}\"")
            else:
                raise RuntimeError(f"event trigger failed: the \"{event_.name}\" event does not support initial state transitions")
        
        transition = Transition(state_machine=self, event=event_, source=self.state, target=event_.target)
        return await transition(*args, **kwargs)
    
    async def enter_state(self, state: Union[State, StateEnum, str], *args, **kwargs) -> bool:
        """Enter a state without triggering an event.
        
        This method can be used to establish an initial state as an alternative to the object initializer,
        which cannot run actions as it is not a coroutine.
        
        When a state is directly entered, the entry and exit actions are executed on the state machine and the
        source and target states involved. Directly entering a state outside of establishing an initial state
        is an atypical operation that should be avoided without very specific motivations as it can lead to
        inconsistent object state managed via event actions.
        
        # TODO: Forbid via a config class attribute?
        
        Args:
            state: The state to enter.
            args: Supplemental positional arguments to be passed to the transition and triggered actions.
            kwargs: Supplemental keyword arguments to be passed to the transition and triggered actions.
        
        Returns:
            A boolean value indicating if the transition was successful.
        
        Raises:
            ValueError: Raised if the state object is not a part of the state machine.
            LookupError: Raised if the state cannot be found by name or enum value.
            TypeError: Raised if the state value given is not a State, StateEnum, or str object.
        """
        if isinstance(state, State):
            state_ = state
            if state_ not in self._states:
                raise ValueError(f"state entry failed: the state object given is not in the state machine")
        elif isinstance(state, (StateEnum, str)):
            name = state.name if isinstance(state, StateEnum) else state
            state_ = self.get_state(name)
            if not state_:
                raise LookupError(f"state entry failed: no state was found with the name \"{name}\"")
        else:
            raise TypeError(f"state entry failed: unexpected value of type \"{state.__class__.__name__}\": {state}")
        
        # Run a state transition. Since there is no event, only the state enter/exit actions are triggered
        transition = Transition(state_machine=self, source=self.state, target=state_)
        return await transition(*args, **kwargs)
    
    ##
    # Actions
    
    async def guard_transition(self, transition: 'Transition', *args, **kwargs) -> bool:
        """Guard the execution of every transition in the state machine.
        
        Guard actions can cancel the execution of transitions by returning `False` or 
        raising an `AssertionError`.
        
        This method is provided for subclasses to override.
        
        Args:
            transition: The transition being applied to the state machine.
            args: A list of supplemental positional arguments passed when the transition was triggered.
            kwargs: A dict of supplemental keyword arguments passed when the transition was triggered.
        
        Returns:
            A boolean that indicates if the transition should be allowed to proceed.
        
        Raises:
            AssertionError: Raised to fail the guard exceptionally
        """
        return True
    
    async def before_transition(self, transition: 'Transition', *args, **kwargs) -> None:
        """Run before every transition in the state machine.
        
        This method is provided for subclasses to override.
        
        Args:
            transition: The transition being applied to the state machine.
            args: A list of supplemental positional arguments passed when the transition was triggered.
            kwargs: A dict of supplemental keyword arguments passed when the transition was triggered.
        """
        pass
    
    async def on_transition(self, transition: 'Transition', *args, **kwargs) -> None:
        """Run on every state change in the state machine.
        
        On actions are run at the moment that the state changes within the state machine,
        before any action defined on the states and event involved in the transition.
        
        This method is provided for subclasses to override.
        
        Args:
            transition: The transition being applied to the state machine.
            args: A list of supplemental positional arguments passed when the transition was triggered.
            kwargs: A dict of supplemental keyword arguments passed when the transition was triggered.
        """
        pass
    
    async def after_transition(self, transition: 'Transition', *args, **kwargs) -> None:
        """Run after every transition in the state machine.
        
        This method is provided for subclasses to override.
        
        Args:
            transition: The transition being applied to the state machine.
            args: A list of supplemental positional arguments passed when the transition was triggered.
            kwargs: A dict of supplemental keyword arguments passed when the transition was triggered.
        """
        pass
    
    class Config:
        # direct_entry?
        allow_entry = "initial" # any, forbid, accept...
        guard_with = "exception" # warning, silence


class Transition(pydantic.BaseModel):
    """Transition objects model a state change within a state machine.
    
    Args:
        state_machine: The state machine in which the transition is occurring.
        source: 
        target: ...
        event: ...
    
    Attributes:        
        state_machine: The state machine in which the transition is occurring.
        source: The state of the state machine when the transition started. None indicates an initial state.
        target: The state that the state machine will be in once the transition has finished.
        event: The event that triggered the transition. None indicates that the state was entered directly.
        created_at: When the transition was created.
        started_at: When the transition started. None if the transition has not been called.
        finished_at: When the transition finished. None if the transition has not been called or is underway.
        cancelled: Whether or not the transition was cancelled by a guard callback or action. None if the transition has not been called or is underway.
        args: Supplemental positional arguments passed to the transition when it was called.
        kwargs: Supplemental keyword arguments passed to the transition when it was called.
    """    
    state_machine: StateMachine
    source: Optional[State] = None
    target: State
    event: Optional[Event] = None
    created_at: datetime.datetime = pydantic.Field(default_factory=datetime.datetime.now)
    started_at: Optional[datetime.datetime] = None
    finished_at: Optional[datetime.datetime] = None
    cancelled: Optional[bool] = None
    args: Optional[List[Any]] = None
    kwargs: Optional[Dict[str, Any]] = None
    
    def __init__(self, state_machine: StateMachine, *args, **kwargs) -> None:
        super().__init__(state_machine=state_machine, *args, **kwargs)
        self.state_machine = state_machine # Ensure we have a reference and not a copy (Pydantic behavior)
    
    async def __call__(self, *args, **kwargs) -> bool:
        """Execute the transition."""
        if self.started_at:
            raise RuntimeError(f"transition has already been executed")
        
        self.args = args
        self.kwargs = kwargs
        
        async with self._lifecycle():                        
            # Guards can cancel the transition via return value or failed assertion
            self.cancelled = False
            try:
                if not await _call_with_matching_parameters(self.state_machine.guard_transition, self, *args, **kwargs):
                    raise AssertionError(f"transition cancelled by guard_transition callback")
            except AssertionError:
                self.cancelled = True
                return False            
            await _call_with_matching_parameters(self.state_machine.before_transition, self, *args, **kwargs)

            try:
                results = await self._run_actions(self.event, Action.Types.guard)
                success = (
                    functools.reduce(lambda x, y: x and y, results, True) if results
                    else True
                )
                if not success:
                    raise AssertionError(f"transition cancelled by guard action")
            except AssertionError:
                self.cancelled = True
                return False
            await self._run_actions(self.event, Action.Types.before)
                
            # Switch between states and try to stay consistent. Actions can be lost in failures
            if not self.is_internal:
                try:
                    await self._run_actions(self.source, Action.Types.exit)
                    self.state_machine._state = self.target
                    await _call_with_matching_parameters(self.state_machine.on_transition, self, *args, **kwargs)
                    await self._run_actions(self.event, Action.Types.on)
                    await self._run_actions(self.target, Action.Types.entry)
                except Exception:
                    self.state_machine._state = self.source
                    raise
                    
            await self._run_actions(self.event, Action.Types.after)
            await _call_with_matching_parameters(self.state_machine.after_transition, self, *args, **kwargs)
            
            return True
    
    # TODO: This is actually a self_transition and internal is where target is None
    # TODO: self transition will call the entry/exit callbacks while internal will not
    @property
    def is_internal(self) -> bool:
        """Return a boolean value that indicates if the source and target states are the same."""
        return self.source == self.target
    
    @property
    def is_executing(self) -> bool:
        """Return a boolean value that indicates if the transition is in progress."""
        return self.started_at is not None and self.finished_at is None
    
    @property
    def is_finished(self) -> bool:
        """Return a boolean value that indicates if the transition has finished."""
        return self.started_at is not None and self.finished_at is not None

    @property
    def runtime(self) -> Optional[datetime.timedelta]:
        """Return a time delta value detailing how long the transition took to execute."""
        return self.finished_at - self.started_at if self.is_finished else None

    @contextlib.asynccontextmanager
    async def _lifecycle(self):
        """Manage lifecycle context for transition execution."""
        try:
            self.started_at = datetime.datetime.now()
            yield
            
        finally:
            self.finished_at = datetime.datetime.now()
    
    async def _run_actions(self, model: Optional[BaseModel], type_: Action.Types) -> Optional[List[Any]]:
        """Run all the actions of a given type attached to a State or Event model.
        
        Returns:
            An aggregated list of return values from the actions run or None if the model is None.
        """
        return await model._run_actions(type_, transition=self, *self.args, **self.kwargs) if model else None

StateIdentifier = Union[StateEnum, str]
Source = Union[None, StateIdentifier, List[StateIdentifier], Type[StateEnum]]
Target = Union[None, StateIdentifier]

class EventDescriptor(pydantic.BaseModel):
    description: Optional[str] = None
    source: List[Union[None, StateIdentifier]]
    target: Target
    guard: List[Callable]
    before: List[Callable]
    on: List[Callable]
    after: List[Callable]
    
    @pydantic.validator("source", pre=True)
    @classmethod
    def _listify_sources(cls, value: Source) -> List[Union[None, StateIdentifier]]:
        identifiers = []
        
        if isinstance(value, list):
            identifiers.extend(value)
        else:
            identifiers.append(value)
        
        return identifiers

    @pydantic.validator("source", each_item=True, pre=True)
    def _map_enums(cls, v) -> Optional[str]:
        if isinstance(v, StateEnum):
            return v.name
        
        return v
    
    @pydantic.validator("guard", "before", "on", "after", pre=True)
    @classmethod
    def _listify_actions(cls, value: Union[None, Callable, List[Callable]]) -> List[Callable]:
        callables = []
        
        if value is None:
            pass
        elif isinstance(value, Callable):
            callables.append(value)
        elif isinstance(value, list):
            callables.extend(value)
        
        return callables

class ActionDescriptor(pydantic.BaseModel):
    model: Type[BaseModel]
    name: str
    description: Optional[str] = None
    type: Action.Types
    callable: Callable
    
# TODO: Make ... an alias for "any"?
# TODO: Internal transition: target is blank, doesn't change
# TODO: A default action without specifying source or target will create a universal internal action for whatever state
def event(
    description: Optional[str] = None, 
    source: Source = ..., 
    target: Optional[Target] = None,
    *,
    guard: Union[None, Callable, List[Callable]] = None,
    before: Union[None, Callable, List[Callable]] = None, 
    after: Union[None, Callable, List[Callable]] = None,
    **kwargs
) -> None:
    """Transform a method into a state machine event."""      
    def decorator(fn):
        target_ = target.name if isinstance(target, StateEnum) else target
        descriptor = EventDescriptor(
            description=description,
            source=source,
            target=target_,
            guard=guard,
            before=before,
            after=after,
            on=fn
        )
        
        @functools.wraps(fn)
        async def event_trigger(self, *args, **kwargs) -> bool:
            # NOTE: The original function is attached as an on event handler
            return await self.trigger(fn.__name__, *args, **kwargs)
        
        event_trigger.__event_descriptor__ = descriptor
        return event_trigger

    return decorator
    
def enter_state(name: Union[str, StateEnum], description: str = "") -> None:
    """Transform a method into an enter state action."""
    return _state_action(name, Action.Types.entry, description)

def exit_state(name: Union[str, StateEnum], description: str = "") -> None:
    """Transform a method into an exit state action."""
    return _state_action(name, Action.Types.exit, description)

def _state_action(name: Union[str, StateEnum], type_: Action.Types, description: str = ""):
    def decorator(fn):
        name_ = name.name if isinstance(name, StateEnum) else name
        descriptor = ActionDescriptor(
            model=State,
            name=name_,
            description=description,
            type=type_,
            callable=fn
        )
        
        fn.__action_descriptor__ = descriptor
        return fn

    return decorator

def guard_event(name: str, description: str = "") -> None:
    """Transform a method into a before event action."""
    return _event_action(name, Action.Types.guard, description)
    
def before_event(name: str, description: str = "") -> None:
    """Transform a method into a before event action."""
    return _event_action(name, Action.Types.before, description)

def after_event(name: str, description: str = "") -> None:
    """Transform a method into an after event action."""
    return _event_action(name, Action.Types.after, description)

def _event_action(name: str, type_: Action.Types, description: str = ""):
    def decorator(fn):
        descriptor = ActionDescriptor(
            model=Event,
            name=name,
            description=description,
            type=type_,
            callable=fn
        )
        
        fn.__action_descriptor__ = descriptor
        return fn

    return decorator

def _summarize(
    values: Sequence[str], *, conjunction: str = "and", quote=False, oxford_comma: bool = True
) -> str:
    """Concatenate a sequence of strings into a series suitable for use in English output.

    Items are joined using a comma and a configurable conjunction, defaulting to 'and'.
    """
    count = len(values)
    values = _quote(values) if quote else values
    if count == 0:
        return ""
    elif count == 1:
        return values[0]
    elif count == 2:
        return f" {conjunction} ".join(values)
    else:
        series = ", ".join(values[0:-1])
        last_item = values[-1]
        delimiter = "," if oxford_comma else ""
        return f"{series}{delimiter} {conjunction} {last_item}"

def _quote(values: Sequence[str]) -> List[str]:
    """Return a sequence of strings surrounding each value in double quotes."""
    return list(map(lambda v: f"\"{v}\"", values))
    
def _parameters_matching_signature(signature: inspect.Signature, *args, **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
    """Return a tuple of positional and keyword parameters that match a callable signature.
    
    This function reduces input parameters down to the subset that matches the given signature.
    It supports callback based APIs by allowing each callback to opt into the parameters of interest
    by including them in the function signature. The matching subset of parameters returned may be
    insufficient for satisfying the signature but will not contain extraneous non-matching parameters.
    """
    parameters: Mapping[
        str, inspect.Parameter
    ] = dict(
        filter(
            lambda item: item[0] not in {"self", "cls"},
            signature.parameters.items()
        )
    )
    
    args_copy, kwargs_copy = collections.deque(args), kwargs.copy()
    matched_args, matched_kwargs = [], {}    
    for name, parameter in parameters.items():
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            # positional only: must be dequeued args
            if len(args_copy):
                matched_args.append(args_copy.popleft())
                
        elif parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            # positional or keyword: check kwargs first and then dequeue from args
            if name in kwargs_copy:
                matched_kwargs[name] = kwargs_copy.pop(name)
                if len(args_copy):
                    # pop the positional arg we have consumed via keyword match
                    args_copy.popleft()
            elif len(args_copy):
                matched_args.append(args_copy.popleft())
                
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            matched_args.extend(args_copy)
            args_copy.clear()
            
        elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            if name in kwargs_copy:
                matched_kwargs[name] = kwargs_copy.pop(name)
                
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            matched_kwargs.update(kwargs_copy)
            kwargs_copy.clear()
    
    return matched_args, matched_kwargs

async def _call_with_matching_parameters(callable: Callable, *args, **kwargs) -> Any:
    """Call a callable with all parameters that match its signature and return the results."""
    matched_args, matched_kwargs = _parameters_matching_signature(inspect.Signature.from_callable(callable), *args, **kwargs)
    if asyncio.iscoroutinefunction(callable):
        return await callable(*matched_args, **matched_kwargs)
    else:
        return callable(*matched_args, **matched_kwargs)
