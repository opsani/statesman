import asyncio
import datetime
import builtins
import contextlib
from typing import Any, AsyncIterator, Dict, Iterator, Optional, List
import pydantic
import pytest
import statesman

import devtools

builtins.debug = devtools.debug

pytestmark = pytest.mark.asyncio

class TestBaseModel:
    @pytest.fixture
    def model(self) -> statesman.BaseModel:
        return statesman.BaseModel()

    @pytest.fixture
    def actions(self) -> List[statesman.Action]:
        return [
            statesman.Action(callable=lambda: 1234, type=statesman.Action.Types.entry),
            statesman.Action(callable=lambda: 1234, type=statesman.Action.Types.after),
            statesman.Action(callable=lambda: 1234, type=None),
            statesman.Action(callable=lambda: 5678, type=statesman.Action.Types.after),
            statesman.Action(callable=lambda: "whatever"),
        ]
    
    def test_add_action(self, model: statesman.BaseModel) -> None:
        assert model._actions == []
        action = statesman.Action(callable=lambda: 1234, type=statesman.Action.Types.entry)
        model._add_action(action)
        assert model._actions == [action]
    
    def test_remove_action(self, model: statesman.BaseModel) -> None:
        assert model._actions == []
        action = statesman.Action(callable=lambda: 1234, type=statesman.Action.Types.entry)
        model._add_action(action)
        assert model._actions == [action]
        model._remove_action(action)
        assert model._actions == []    
    
    class TestCollection:
        @pytest.fixture
        def model(self, actions: List[statesman.Action]) -> statesman.BaseModel:
            model = statesman.BaseModel()
            model._actions = actions.copy()
            return model
        
        def test_by_object(self, model: statesman.BaseModel, actions: List[statesman.Action]) -> None:
            assert model._actions == actions
            action = actions[4]
            model._remove_actions([action])
            assert model._actions == actions[0:4]
        
        def test_none(self, model: statesman.BaseModel, actions: List[statesman.Action]) -> None:
            assert model._actions == actions            
            model._remove_actions()
            assert model._actions == []
        
        def test_type(self, model: statesman.BaseModel, actions: List[statesman.Action]) -> None:
            assert model._actions == actions            
            model._remove_actions(statesman.Action.Types.entry)
            assert model._actions == actions[1:5]
    
        def test_get_actions(self, model: statesman.BaseModel, actions: List[statesman.Action]) -> None:
            assert model._actions == actions
            matched = model._get_actions(statesman.Action.Types.after)
            assert matched == [actions[1], actions[3]]
    
class TestState:    
    class States(statesman.StateEnum):
        first = "First"
        second = "Second"
        
    @pytest.fixture
    def state(self) -> statesman.State:
        return statesman.State(name="Testing")
    
    def test_add_action(self, state: statesman.State) -> None:
        action = state.add_action(lambda: 1234, statesman.Action.Types.entry)
        assert action
        assert state.actions == [action]
    
    def test_add_action_invalid_type(self, state: statesman.State) -> None:
        with pytest.raises(ValueError, match='cannot add state action with type "after": must be "entry" or "exit"'):
            state.add_action(lambda: 1234, statesman.Action.Types.after)
    
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (States.first, True),
            ("first", True),
            ("First", False),
            (1234, False),
            (None, False),
        ],
    )
    def test_equality(self, value: Any, expected: bool) -> None:
        state = statesman.State(name=TestState.States.first)
        assert (state == value) == expected
    
    class TestListFrom:
        def test_enum(self) -> None:
            states = statesman.State.from_enum(States)
            assert states
            assert len(states) == 4
            assert (states[0].name, states[0].description) == ("starting", "Starting")

class TestAction:
    def test_callable_is_required(self) -> None:
        with pytest.raises(pydantic.ValidationError) as e:
            statesman.Action()
        
        assert e
        assert "2 validation errors for Action" in str(e.value)
        assert e.value.errors()[0]["loc"] == ("callable",)
        assert (
            e.value.errors()[0]["msg"]
            == 'field required'
        )
        
    def test_signature_is_hydrated(self) -> None:
        def some_func(count: int, labels: Dict[str, str]) -> float:
            ...
            
        action = statesman.Action(callable=some_func)
        assert action.signature
        assert repr(action.signature) == "<Signature (count: int, labels: Dict[str, str]) -> float>"
    
    def test_types(self) -> None:
        action = statesman.Action(callable=lambda: 1234, type=statesman.Action.Types.entry)
        assert action.type == "entry"
    
    async def test_call_action(self) -> None:
        action = statesman.Action(callable=lambda: 1234)
    
    async def test_argument_matching(self) -> None:
        # TODO: Test with and without *args and **kwargs
        def action_body(count: int, another: bool = False, *args, something = None, number = 1234) -> None:
            ...
            
        # parametrize with a variations of args
        # TODO: Test passing count as positional or keyword, another as keyword while count is positional
        # TODO: Test signature with and without *args and **kwargs
        action = statesman.Action(callable=action_body)
        await action(1234)

class States(statesman.StateEnum):
    starting = "Starting"
    running = "Running"
    stopping = "Stopping"
    stopped = "Stopped"

class TestStateMachine:
    @pytest.fixture
    def state_machine(self) -> statesman.StateMachine:
        return statesman.StateMachine(states=statesman.State.from_enum(States))
    
    async def test_get_states_names(self, state_machine: statesman.StateMachine) -> None:
        states = state_machine.get_states("starting", "stopped")
        assert len(states) == 2
        assert list(map(lambda i: i.name, states)) == ["starting", "stopped"]
    
    async def test_get_states_by_state_enum(self, state_machine: statesman.StateMachine) -> None:
        states = state_machine.get_states(States)
        assert len(states) == 4
        assert list(map(lambda i: i.name, states)) == ['starting', 'running', 'stopping', 'stopped']
    
    async def test_get_states_by_state_enum_list(self, state_machine: statesman.StateMachine) -> None:
        states = state_machine.get_states(States.starting, States.running)
        assert len(states) == 2
        assert list(map(lambda i: i.name, states)) == ['starting', 'running']

class TestTransition:
    @pytest.fixture
    def transition(self) -> statesman.Transition:
        state_machine = statesman.StateMachine()
        state_machine.add_states(statesman.State.from_enum(States))
        starting = state_machine.get_state(States.starting)
        stopping = state_machine.get_state(States.stopping)
        
        return statesman.Transition(state_machine=state_machine, source=starting, target=stopping)
    
    async def test_lifecycle(self, transition: statesman.Transition) -> None:        
        assert transition.created_at
        assert transition.started_at is None
        assert transition.finished_at is None
        assert transition.cancelled is None
        await transition()
        assert transition.started_at is not None
        assert transition.finished_at is not None
        assert transition.cancelled == False
    
    async def test_runtime(self, transition: statesman.Transition) -> None:
        assert transition.runtime is None
        await transition()
        assert transition.runtime is not None
        assert isinstance(transition.runtime, datetime.timedelta)
        
    async def test_is_finished(self, transition: statesman.Transition) -> None:
        assert transition.is_finished is False
        await transition()
        assert transition.is_finished is True
    
    async def test_is_executing(self, transition: statesman.Transition) -> None:
        was_executing = None        
        def check_executing(transition: statesman.Transition):
            nonlocal was_executing
            was_executing = transition.is_executing                    
        state = transition.state_machine.get_state(States.stopping)
        state.add_action(check_executing, statesman.Action.Types.entry)
        assert transition.is_executing is False
        await transition()
        assert was_executing is True
        assert transition.is_executing is False
    
    async def test_args_and_kwargs(self, transition: statesman.Transition) -> None:
        assert transition.args is None
        assert transition.kwargs is None
        await transition(1234, foo="Bar")
        assert transition.args == (1234,)
        assert transition.kwargs == {"foo": "Bar"}
    
    async def test_params_passed_to_actions(self, transition: statesman.Transition) -> None:
        called = False
        def check_executing(count: int, foo: str):
            nonlocal called
            called = True
            assert count == 1234
            assert foo == "Bar"
        state = transition.state_machine.get_state(States.stopping)
        state.add_action(check_executing, statesman.Action.Types.entry)
        assert transition.is_executing is False
        await transition(1234, foo="Bar")
        assert called is True # Ensure that our inner assertions actually ran
        
    # TODO: Support the arguments: transition? what else?
    # TODO: Test with garbage args
    
class TestProgrammaticStateMachine:
    def test_add_state(self) -> None:
        state_machine = statesman.StateMachine()
        state_machine.add_state(
            statesman.State(
                name=States.starting
            )
        )
        assert len(state_machine.states) == 1
        state = state_machine.states[0]
        assert state == States.starting
        
    def test_add_state_cannot_duplicate_existing_name(self) -> None:
        state_machine = statesman.StateMachine()
        state_machine.add_state(
            statesman.State(
                name=States.starting
            )
        )
        assert len(state_machine.states) == 1
        
        with pytest.raises(ValueError, match='a state named "starting" already exists'):
            state_machine.add_state(
                statesman.State(
                    name=States.starting
                )
            )
    
    def test_add_states(self) -> None:
        state_machine = statesman.StateMachine()
        state_machine.add_states(statesman.State.from_enum(States))
        assert len(state_machine.states) == 4
        state = state_machine.states[0]
        assert state.name == States.starting.name
        assert state.description == States.starting.value
    
    def test_add_states_enum_names(self) -> None:
        state_machine = statesman.StateMachine()
        state_machine.add_states([
            statesman.State(
                name=States.starting,
                description=States.starting
            ),
            statesman.State(
                name=States.stopping
            )
        ])
        assert len(state_machine.states) == 2
        state1, state2 = state_machine.states
        assert state1.name == "starting"
        assert state1.description == "Starting"
        assert state2.name == "stopping"
        assert state2.description is None # we didn't pass description
    
    def test_enter_states_via_initializer(self) -> None:
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States))
        assert len(state_machine.states) == 4
        state = state_machine.states[0]
        assert state == States.starting
    
    async def test_enter_state_name_not_found(self) -> None:
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States))
        assert state_machine.state == None
        with pytest.raises(LookupError, match='state entry failed: no state was found with the name "invalid"'):
            await state_machine.enter_state("invalid")
        
    async def test_enter_state_enum_not_found(self) -> None:
        class OtherStates(statesman.StateEnum):
            invalid = "invalid"
            
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States))
        assert state_machine.state == None
        with pytest.raises(LookupError, match='state entry failed: no state was found with the name "invalid"'):
            await state_machine.enter_state(OtherStates.invalid)
        
    async def test_enter_state_not_in_machine(self) -> None:
        state = statesman.State("invalid")
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States))
        assert state_machine.state == None
        with pytest.raises(ValueError, match='state entry failed: the state object given is not in the state machine'):
            await state_machine.enter_state(state)
    
    async def test_enter_state_runs_state_actions(self, mocker) -> None:
        state_machine = statesman.StateMachine()
        state_machine.add_states(statesman.State.from_enum(States))
        starting = state_machine.get_state(States.starting)
        stopping = state_machine.get_state(States.stopping)
        
        stub = mocker.stub(name='starting')
        action = lambda: stub()
        starting.add_action(action, statesman.Action.Types.entry)
        starting.add_action(action, statesman.Action.Types.exit)
        stopping.add_action(action, statesman.Action.Types.entry)
        stopping.add_action(action, statesman.Action.Types.exit)
        
        # Test from zero state producing one entry state action
        await state_machine.enter_state(starting)
        stub.assert_called_once()
        
        # Assign a new state producing two additional actions: exit starting, entry stopping
        await state_machine.enter_state(stopping)
        stub.assert_called()
        assert stub.call_count == 3
    
    @pytest.mark.parametrize(
        ("callback"),
        [
            "guard_transition",
            "before_transition",
            "on_transition",
            "after_transition"            
        ],
    )
    async def test_enter_state_with_args(self, callback, mocker) -> None:
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States), state=States.starting)
        assert state_machine.state == States.starting
        
        with extra(state_machine):
            callback_mock = mocker.spy(state_machine, callback)
            await state_machine.enter_state(States.stopping, 1234, foo="bar")
            callback_mock.assert_called_once()
            assert len(callback_mock.call_args.args) == 2
            assert isinstance(callback_mock.call_args.args[0], statesman.Transition), "expected a Transition"
            assert callback_mock.call_args.args[1]
            assert callback_mock.call_args.kwargs == { "foo": "bar" }
    
    async def test_doesnt_run_on_callbacks_for_internal_transitions(self, mocker) -> None:
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States), state=States.starting)
        assert state_machine.state == States.starting
        
        # NOTE: we are already in Starting and entering it again
        with extra(state_machine):
            callback_mock = mocker.spy(state_machine, "on_transition")
            await state_machine.enter_state(States.starting, 1234, foo="bar")
            callback_mock.assert_not_called()
    
    def test_add_event_fails_if_existing(self) -> None:
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States), state=States.starting)
        state = state_machine.states[0]
        state_machine.add_event(
            statesman.Event(
                name="finish",
                sources=[state],
                target=state
            )
        )
        with pytest.raises(ValueError, match='an event named "finish" already exists'):
            state_machine.add_event(
                statesman.Event(
                    name="finish",
                    sources=[state],
                    target=state
                )
            )
        
    def test_add_event_fails_with_unknown_state(self) -> None:
        state_machine = statesman.StateMachine()
        state = statesman.State("invalid")
        with pytest.raises(ValueError, match='cannot add an event that references unknown states: "invalid"'):
            state_machine.add_event(
                statesman.Event(
                    name="finish",
                    sources=[state],
                    target=state
                )
            )
    
    def test_removing_state_clears_all_referencing_events(self) -> None:
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States), state=States.starting)
        state = state_machine.states[0]
        event = statesman.Event(
            name="finish",
            sources=[state],
            target=state
        )
        state_machine.add_event(event)
        assert state_machine.events == [event]
        
        state_machine.remove_state(state)
        assert state_machine.events == []
            
    class TestTrigger:
        @pytest.fixture
        def state_machine(self) -> statesman.StateMachine:
            state_machine = statesman.StateMachine()
            state_machine.add_states([
                statesman.State(
                    name=States.starting
                ),
                statesman.State(
                    name=States.stopping
                )
            ])
            state_machine.add_event(
                statesman.Event(
                    name="finish",
                    sources=state_machine.get_states(States.starting, States.running),
                    target=state_machine.get_state(States.stopping)
                )
            )
            return state_machine        
        
        async def test_get_event(self, state_machine: statesman.StateMachine) -> None:
            event = state_machine.get_event("finish")
            assert event is not None
        
        async def test_get_event_not_found(self, state_machine: statesman.StateMachine) -> None:
            event = state_machine.get_event("invalid")
            assert event is None
        
        async def test_get_event_invalid_type_raises(self, state_machine: statesman.StateMachine) -> None:
            assert state_machine.state == None
            with pytest.raises(TypeError) as e:
                state_machine.get_event(1234)
            
            assert str(e.value) == "cannot get event for name of type \"int\": 1234"
                    
        async def test_by_name(self, state_machine: statesman.StateMachine) -> None:
            await state_machine.enter_state(States.starting)
            assert state_machine.state == States.starting
            await state_machine.trigger("finish")
            assert state_machine.state == States.stopping
            
        async def test_by_event(self, state_machine: statesman.StateMachine) -> None:
            await state_machine.enter_state(States.starting)
            assert state_machine.state == States.starting
            event = state_machine.get_event("finish")
            await state_machine.trigger(event)
            assert state_machine.state == States.stopping
        
        async def test_trigger_without_state_raises(self, state_machine: statesman.StateMachine) -> None:
            assert state_machine.state == None
            with pytest.raises(RuntimeError, match='event trigger failed: the "finish" event does not support initial state transitions'):
                await state_machine.trigger("finish")
        
        async def test_trigger_from_incompatible_state(self, state_machine: statesman.StateMachine) -> None:
            await state_machine.enter_state(States.stopping)
            with pytest.raises(RuntimeError, match='event trigger failed: the "finish" event cannot be triggered from the current state of "stopping"'):
                await state_machine.trigger("finish")
            
        async def test_with_invalid_name(self, state_machine: statesman.StateMachine) -> None:
            await state_machine.enter_state(States.starting)
            with pytest.raises(LookupError, match="event trigger failed: no event was found with the name \"invalid\""):
                await state_machine.trigger("invalid")
        
        async def test_with_invalid_type(self, state_machine: statesman.StateMachine) -> None:
            await state_machine.enter_state(States.starting)
            with pytest.raises(TypeError, match="event trigger failed: cannot trigger an event of type \"int\": 1234"):
                await state_machine.trigger(1234)
        
        async def test_with_event_not_in_machine(self, state_machine: statesman.StateMachine) -> None:
            invalid_event = statesman.Event(
                name="invalid",
                sources=state_machine.states,
                target=state_machine.get_state(States.stopping)
            )
            await state_machine.enter_state(States.starting)
            with pytest.raises(TypeError, match="event trigger failed: cannot trigger an event of type \"int\": 1234"):
                await state_machine.trigger(1234)
    
        async def test_cancel_via_guard_state_machine_method(self, state_machine: statesman.StateMachine, mocker) -> None:
            await state_machine.enter_state(States.starting)
            with extra(state_machine):
                guard_mock = mocker.patch.object(state_machine, "guard_transition")
                guard_mock.return_value = False
                success = await state_machine.trigger("finish")
                guard_mock.assert_awaited_once()
                assert not success, "should have been guarded"
        
        async def test_non_assertion_errors_raise(self, state_machine: statesman.StateMachine, mocker) -> None:
            await state_machine.enter_state(States.starting)
            with extra(state_machine):
                guard_mock = mocker.patch.object(state_machine, "guard_transition")
                guard_mock.side_effect = RuntimeError(f"failed!")
                
                with pytest.raises(RuntimeError, match="failed!"):
                    success = await state_machine.trigger("finish")
                    guard_mock.assert_awaited_once()
                    assert not success, "should have been guarded"
        
        class TestActions:
            async def test_guard(self, state_machine: statesman.StateMachine, mocker) -> None:
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event("finish")
                guard_action = mocker.stub(name='action')
                event.add_action(lambda: guard_action(), statesman.Action.Types.guard)
                assert await state_machine.trigger("finish"), "guard passed"
                guard_action.assert_called_once()
            
            async def test_cancel_via_guard_action_bool(self, state_machine: statesman.StateMachine, mocker) -> None:                
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event("finish")
                guard_action = mocker.stub(name='action')
                def cancel() -> bool:
                    guard_action()
                    return False
                
                event.add_action(lambda: cancel(), statesman.Action.Types.guard)
                # NOTE: The AssertionError is being caught and aborts the test
                success = await state_machine.trigger("finish")
                guard_action.assert_called_once()
                assert not success, "should have been cancelled by guard"
                
            async def test_cancel_via_guard_action_exception(self, state_machine: statesman.StateMachine, mocker) -> None:                
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event("finish")
                guard_action = mocker.stub(name='action')
                def cancel() -> None:
                    guard_action()
                    raise AssertionError("should be suppressed")
                
                event.add_action(lambda: cancel(), statesman.Action.Types.guard)
                # NOTE: The AssertionError is being caught and aborts the test
                success = await state_machine.trigger("finish")
                assert not success, "cancelled by guard"
                guard_action.assert_called_once()
                
            async def test_before(self, state_machine: statesman.StateMachine, mocker) -> None:
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event("finish")
                before_action = mocker.stub(name='action')
                event.add_action(lambda: before_action(), statesman.Action.Types.before)                
                await state_machine.trigger("finish")
                before_action.assert_called_once()
            
            async def test_after(self, state_machine: statesman.StateMachine, mocker) -> None:
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event("finish")
                after_action = mocker.stub(name='action')
                event.add_action(lambda: after_action(), statesman.Action.Types.after)
                await state_machine.trigger("finish")
                after_action.assert_called_once()
            
            async def test_on(self, state_machine: statesman.StateMachine, mocker) -> None:
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event("finish")
                on_action = mocker.stub(name='action')
                event.add_action(lambda: on_action(), statesman.Action.Types.on)
                await state_machine.trigger("finish")
                on_action.assert_called_once()
            
            async def test_inheritable_actions(self, state_machine: statesman.StateMachine, mocker) -> None:
                with extra(state_machine):
                    guard_transition = mocker.spy(state_machine, "guard_transition")
                    before_transition = mocker.spy(state_machine, "before_transition")
                    on_transition = mocker.spy(state_machine, "on_transition")
                    after_transition = mocker.spy(state_machine, "after_transition")
                    await state_machine.enter_state(States.starting)
                    await state_machine.trigger("finish")
                    
                    # From None -> Starting, Starting -> Stopped
                    guard_transition.assert_called()
                    assert guard_transition.call_count == 2
                    before_transition.assert_called()
                    assert before_transition.call_count == 2
                    on_transition.assert_called()
                    assert on_transition.call_count == 2
                    after_transition.assert_called()
                    assert after_transition.call_count == 2

class TestDecoratorStateMachine:
    class ProcessStates(statesman.StateEnum):
        starting = "Starting..."
        running = "Running..."
        stopping = "Stopping..."
        stopped = "Terminated."
    
    def test_set_states_declaratively(self) -> None:
        class TestMachine(statesman.StateMachine):
            __state__: TestDecoratorStateMachine.ProcessStates = TestDecoratorStateMachine.ProcessStates.stopping
        
        state_machine = TestMachine()
        assert len(state_machine.states) == 4
        assert state_machine.states[0] == States.starting
        assert isinstance(state_machine.state, statesman.State)
        assert state_machine.state == TestDecoratorStateMachine.ProcessStates.stopping
    
    def test_set_initial_state_declaratively(self) -> None:
        class TestMachine(statesman.StateMachine):
            __state__: States = States.stopping
        
        state_machine = TestMachine()
        assert state_machine.state == States.stopping
    
    def test_set_initial_state_declaratively_optional(self) -> None:
        class TestMachine(statesman.StateMachine):
            __state__: Optional[States] = None
        
        state_machine = TestMachine()
        assert state_machine.state is None
    
    def test_set_states_embedded_enum(self) -> None:
        class TestMachine(statesman.StateMachine):
            class States(statesman.StateEnum):
                one = "One"
                two = "Two"
                three = "Three"
                four = "Four"
        
        state_machine = TestMachine()
        assert state_machine.state is None
        assert len(state_machine.states) == 4
        assert state_machine.states[0] == TestMachine.States.one
        assert state_machine.state is None
        
    def test_overrides(self):        
        class TestMachine(statesman.StateMachine):
            _state: Optional[States] = States.stopping
            _initial = pydantic.PrivateAttr(States.starting)
            _state_enum: Optional[States] = States.starting
            foo = 1234
        
        state_machine = TestMachine()
        debug(state_machine, state_machine._state, state_machine._initial, state_machine._state_enum, TestMachine.states, state_machine.states)

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
        async def start(self, command: str = "...") -> None:
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
        
        @statesman.enter_state(States.stopped)
        async def _on_stop(self) -> None:
            self.logs.append(f"_on_stop triggered")
        
        @statesman.after_event("terminate")
        async def _terminated(self) -> None:
            self.logs.append(f"_terminated")
        
        async def after_transition(self, transition: statesman.Transition) -> None:
            if transition.event and transition.event.name == "stop":
                await self.terminate()
    
    @pytest.fixture
    def state_machine(self) -> statesman.StateMachine:
        return TestDecoratorStateMachine.ProcessLifecycle()
        
    async def test_states_added(self, state_machine: statesman.StateMachine) -> None:
        assert len(state_machine.states) == 4
        assert state_machine.states[0].name == "starting"
        assert state_machine.states[0].description == "Starting..."
    
    async def test_events_added(self, state_machine: statesman.StateMachine) -> None:
        event = state_machine.get_event("start")
        assert event
        assert event.description == "Start a Process"
        assert event.sources == [None]
        assert event.target == States.starting
    
    async def test_trigger_event_through_method_call(self, state_machine: statesman.StateMachine) -> None:
        assert state_machine.state is None
        await state_machine.start()
        assert state_machine.state
        assert state_machine.state == States.starting
    
    async def test_trigger_event_through_method_call_with_args(self, state_machine: statesman.StateMachine, mocker) -> None:
        with extra(state_machine):
            callback_mock = mocker.spy(state_machine, "on_transition")
            await state_machine.start(31337, this="That")
            callback_mock.assert_called_once()
            assert 31337 in callback_mock.call_args.args
            assert { "this": "That" } == callback_mock.call_args.kwargs
    
    async def test_process_lifecycle(self, state_machine: statesman.StateMachine, mocker) -> None:
        assert state_machine.pid is None
        assert state_machine.command is None
        
        await state_machine.start("ls -al")
        assert state_machine.command == "ls -al"
        assert state_machine.pid == 31337
        assert state_machine.state == States.starting
        assert len(state_machine.logs) == 0
        
        await state_machine.run()
        assert state_machine.logs == ['Process pid 31337 is now running (command="ls -al")']
        
        await state_machine.stop()
        assert state_machine.logs == [
            'Process pid 31337 is now running (command="ls -al")',
            'Shutting down pid 31337 (command="ls -al")',
            'Terminated pid 31337 ("ls -al")',
            '_on_stop triggered',
            '_terminated'
        ]
        
        # Let the runloop cycle
        await asyncio.sleep(0.001)
        assert state_machine.state == States.stopped
        assert state_machine.pid is None
        assert state_machine.command is None

class TestInitialState:
    class StateMachine(statesman.StateMachine):
        class States(statesman.StateEnum):
            starting = "Starting..."
            running = "Running..."
            stopping = "Stopping..."
            stopped = statesman.InitialState("Terminated.")
                
    def test_initial_property(self) -> None:
        assert TestInitialState.StateMachine.States.__initial__
        assert TestInitialState.StateMachine.States.__initial__ == TestInitialState.StateMachine.States.stopped
    
    def test_initial_state_is_set_on_state_machine(self) -> None:        
        state_machine = TestInitialState.StateMachine()
        assert state_machine.state == TestInitialState.StateMachine.States.stopped
    
    def test_initial_state_can_be_overridden(self) -> None:
        state_machine = TestInitialState.StateMachine(state=TestInitialState.StateMachine.States.running)
        assert state_machine.state == TestInitialState.StateMachine.States.running
    
    def test_cannot_set_multiple_initial_states(self) -> None:
        with pytest.raises(ValueError, match='cannot declare more than one initial state: "States.one" already declared'):
            class States(statesman.StateEnum):
                one = statesman.InitialState("1")
                two = "2"
                three = "3"
                four = statesman.InitialState("4")

@contextlib.contextmanager
def extra(
    obj: pydantic.BaseModel, extra: pydantic.Extra = pydantic.Extra.allow
) -> Iterator[pydantic.BaseModel]:
    """Temporarily override the value of the `extra` setting on a Pydantic object.
    
    Used in tests to support object mocking/spying that relies on setattr to inject mocks.
    """
    original = obj.__config__.extra
    obj.__config__.extra = extra
    try:
        yield obj
    finally:
        obj.__config__.extra = original


async def test_matching_signature_overlapping_params() -> None:
    def some_function(transition: str, *args, **kwargs) -> None:
        ...
    
    args = ("whatever", )
    kwargs = {"transition": "foo"}
    
    await statesman._call_with_matching_parameters(some_function, *args, **kwargs)
    