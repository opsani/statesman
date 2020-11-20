import asyncio
import builtins
import contextlib
import datetime
import inspect
from typing import Any, Coroutine, Dict, Iterator, List, Optional

import devtools
import pydantic
import pytest
import statesman

builtins.debug = devtools.debug

pytestmark = pytest.mark.asyncio


class TestBaseModel:
    @pytest.fixture()
    def model(self) -> statesman.BaseModel:
        return statesman.BaseModel()

    @pytest.fixture()
    def actions(self) -> List[statesman.Action]:
        return [
            statesman.Action(callable=lambda: 1234, type=statesman.Action.Types.entry),
            statesman.Action(callable=lambda: 1234, type=statesman.Action.Types.after),
            statesman.Action(callable=lambda: 1234, type=None),
            statesman.Action(callable=lambda: 5678, type=statesman.Action.Types.after),
            statesman.Action(callable=lambda: 'whatever'),
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
        @pytest.fixture()
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
        first = 'First'
        second = 'Second'

    @pytest.fixture()
    def state(self) -> statesman.State:
        return statesman.State(name='Testing')

    def test_add_action(self, state: statesman.State) -> None:
        action = state.add_action(lambda: 1234, statesman.Action.Types.entry)
        assert action
        assert state.actions == [action]

    def test_add_action_invalid_type(self, state: statesman.State) -> None:
        with pytest.raises(ValueError, match='cannot add state action with type "after": must be "entry" or "exit"'):
            state.add_action(lambda: 1234, statesman.Action.Types.after)

    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            (States.first, True),
            ('first', True),
            ('First', False),
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
            assert (states[0].name, states[0].description) == ('starting', 'Starting')


class TestAction:
    def test_callable_is_required(self) -> None:
        with pytest.raises(pydantic.ValidationError) as e:
            statesman.Action()

        assert e
        assert '2 validation errors for Action' in str(e.value)
        assert e.value.errors()[0]['loc'] == ('callable',)
        assert (
            e.value.errors()[0]['msg']
            == 'field required'
        )

    def test_signature_is_hydrated(self) -> None:
        def some_func(count: int, labels: Dict[str, str]) -> float:
            ...

        action = statesman.Action(callable=some_func)
        assert action.signature
        assert repr(action.signature) == '<Signature (count: int, labels: Dict[str, str]) -> float>'

    def test_types(self) -> None:
        action = statesman.Action(callable=lambda: 1234, type=statesman.Action.Types.entry)
        assert action.type == 'entry'

    async def test_call_action(self) -> None:
        action = statesman.Action(callable=lambda: 1234)

    async def test_argument_matching(self) -> None:
        # TODO: Test with and without *args and **kwargs
        def action_body(count: int, another: bool = False, *args, something=None, number=1234) -> None:
            ...

        # parametrize with a variations of args
        # TODO: Test passing count as positional or keyword, another as keyword while count is positional
        # TODO: Test signature with and without *args and **kwargs
        action = statesman.Action(callable=action_body)
        await action(1234)


class States(statesman.StateEnum):
    starting = 'Starting'
    running = 'Running'
    stopping = 'Stopping'
    stopped = 'Stopped'


class TestStateMachine:
    @pytest.fixture()
    def state_machine(self) -> statesman.StateMachine:
        return statesman.StateMachine(states=statesman.State.from_enum(States))

    async def test_get_states_names(self, state_machine: statesman.StateMachine) -> None:
        states = state_machine.get_states('starting', 'stopped')
        assert len(states) == 2
        assert list(map(lambda i: i.name, states)) == ['starting', 'stopped']

    async def test_get_states_by_state_enum(self, state_machine: statesman.StateMachine) -> None:
        states = state_machine.get_states(States)
        assert len(states) == 4
        assert list(map(lambda i: i.name, states)) == ['starting', 'running', 'stopping', 'stopped']

    async def test_get_states_by_state_enum_list(self, state_machine: statesman.StateMachine) -> None:
        states = state_machine.get_states(States.starting, States.running)
        assert len(states) == 2
        assert list(map(lambda i: i.name, states)) == ['starting', 'running']

    def test_repr(self, state_machine: statesman.StateMachine) -> None:
        assert repr(state_machine) == "StateMachine(states=[State(name='starting', description='Starting'), State(name='running', description='Running'), State(name='stopping', description='Stopping'), State(name='stopped', description='Stopped')], events=[], state=None)"


class TestTransition:
    @pytest.fixture()
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
        await transition(1234, foo='Bar')
        assert transition.args == (1234,)
        assert transition.kwargs == {'foo': 'Bar'}

    async def test_params_passed_to_actions(self, transition: statesman.Transition) -> None:
        called = False

        def check_executing(count: int, foo: str):
            nonlocal called
            called = True
            assert count == 1234
            assert foo == 'Bar'
        state = transition.state_machine.get_state(States.stopping)
        state.add_action(check_executing, statesman.Action.Types.entry)
        assert transition.is_executing is False
        await transition(1234, foo='Bar')
        assert called is True  # Ensure that our inner assertions actually ran

    async def test_internal_transition(self, mocker) -> None:
        state_machine = statesman.StateMachine()
        state_machine.add_states(statesman.State.from_enum(States))
        stopping = state_machine.get_state(States.stopping)
        await state_machine.enter_state(stopping)

        transition = statesman.Transition(
            state_machine=state_machine,
            source=stopping,
            target=stopping,
            state=States.stopping,
            type=statesman.Transition.Types.internal,
        )
        assert transition.state_machine.state == stopping

        entry_stub = mocker.stub(name='entering stopping')
        exit_stub = mocker.stub(name='exiting stopping')
        def entry(): return entry_stub()
        def exit(): return exit_stub()

        stopping.add_action(entry, statesman.Action.Types.entry)
        stopping.add_action(exit, statesman.Action.Types.exit)

        await transition()

        entry_stub.assert_not_called()
        exit_stub.assert_not_called()

    async def test_self_transition(self, mocker) -> None:
        state_machine = statesman.StateMachine()
        state_machine.add_states(statesman.State.from_enum(States))
        stopping = state_machine.get_state(States.stopping)
        await state_machine.enter_state(stopping)

        transition = statesman.Transition(
            state_machine=state_machine,
            source=stopping,
            target=stopping,
            state=States.stopping,
            type=statesman.Transition.Types.self,
        )
        assert transition.state_machine.state == stopping

        entry_stub = mocker.stub(name='entering stopping')
        exit_stub = mocker.stub(name='exiting stopping')
        def entry(): return entry_stub()
        def exit(): return exit_stub()

        stopping.add_action(entry, statesman.Action.Types.entry)
        stopping.add_action(exit, statesman.Action.Types.exit)

        await transition()

        entry_stub.assert_called_once()
        exit_stub.assert_called_once()

    async def test_results_is_none_when_event_is_none(self, transition: statesman.Transition) -> None:
        assert transition.event is None
        assert transition.results is None
        assert await transition()
        assert transition.results is None

    async def test_results_is_populated_with_return_value_of_on_event_handlers(self, mocker) -> None:
        state_machine = statesman.StateMachine()
        state_machine.add_states(statesman.State.from_enum(States))
        stopping = state_machine.get_state(States.stopping)
        starting = state_machine.get_state(States.starting)
        await state_machine.enter_state(stopping)

        event = statesman.Event(
            name='finish',
            sources=[stopping],
            target=stopping,
        )
        start_stub = mocker.stub(name='entering starting')
        start_stub.return_value = 31337
        def on_event(): return start_stub()
        event.add_action(on_event, statesman.Action.Types.on)
        state_machine.add_event(event)

        transition = statesman.Transition(
            state_machine=state_machine,
            source=stopping,
            target=starting,
            type=statesman.Transition.Types.external,
            event=event,
        )
        assert transition.state_machine.state == stopping

        assert await transition()
        assert transition.results == [31337]


class TestProgrammaticStateMachine:
    def test_add_state(self) -> None:
        state_machine = statesman.StateMachine()
        state_machine.add_state(
            statesman.State(
                name=States.starting,
            ),
        )
        assert len(state_machine.states) == 1
        state = state_machine.states[0]
        assert state == States.starting

    def test_add_state_cannot_duplicate_existing_name(self) -> None:
        state_machine = statesman.StateMachine()
        state_machine.add_state(
            statesman.State(
                name=States.starting,
            ),
        )
        assert len(state_machine.states) == 1

        with pytest.raises(ValueError, match='a state named "starting" already exists'):
            state_machine.add_state(
                statesman.State(
                    name=States.starting,
                ),
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
                description=States.starting,
            ),
            statesman.State(
                name=States.stopping,
            ),
        ])
        assert len(state_machine.states) == 2
        state1, state2 = state_machine.states
        assert state1.name == 'starting'
        assert state1.description == 'Starting'
        assert state2.name == 'stopping'
        assert state2.description is None  # we didn't pass description

    def test_enter_states_via_initializer(self) -> None:
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States))
        assert len(state_machine.states) == 4
        state = state_machine.states[0]
        assert state == States.starting

    async def test_enter_state_name_not_found(self) -> None:
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States))
        assert state_machine.state is None
        with pytest.raises(LookupError, match='state entry failed: no state was found with the name "invalid"'):
            await state_machine.enter_state('invalid')

    async def test_enter_state_enum_not_found(self) -> None:
        class OtherStates(statesman.StateEnum):
            invalid = 'invalid'

        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States))
        assert state_machine.state is None
        with pytest.raises(LookupError, match='state entry failed: no state was found with the name "invalid"'):
            await state_machine.enter_state(OtherStates.invalid)

    async def test_enter_state_not_in_machine(self) -> None:
        state = statesman.State('invalid')
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States))
        assert state_machine.state is None
        with pytest.raises(ValueError, match='state entry failed: the state object given is not in the state machine'):
            await state_machine.enter_state(state)

    async def test_enter_state_runs_state_actions(self, mocker) -> None:
        state_machine = statesman.StateMachine()
        state_machine.add_states(statesman.State.from_enum(States))
        starting = state_machine.get_state(States.starting)
        stopping = state_machine.get_state(States.stopping)

        stub = mocker.stub(name='starting')
        def action(): return stub()
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

    async def test_create_no_state(self) -> None:
        state_machine = await statesman.StateMachine.create(states=statesman.State.from_enum(States))
        assert state_machine.state is None

    async def test_create_enter_specific_state(self) -> None:
        state_machine = await statesman.StateMachine.create(
            states=statesman.State.from_enum(States),
            state=States.stopping,
        )
        assert state_machine.state == States.stopping

    @pytest.mark.parametrize(
        ('callback'),
        [
            'guard_transition',
            'before_transition',
            'on_transition',
            'after_transition',
        ],
    )
    async def test_enter_state_with_args(self, callback, mocker) -> None:
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States), state=States.starting)
        assert state_machine.state == States.starting

        with extra(state_machine):
            callback_mock = mocker.spy(state_machine, callback)
            await state_machine.enter_state(States.stopping, 1234, foo='bar')
            callback_mock.assert_called_once()
            assert len(callback_mock.call_args.args) == 2
            assert isinstance(callback_mock.call_args.args[0], statesman.Transition), 'expected a Transition'
            assert callback_mock.call_args.args[1]
            assert callback_mock.call_args.kwargs == {'foo': 'bar'}

    async def test_doesnt_run_state_actions_for_internal_transitions(self, mocker) -> None:
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States), state=States.starting)
        assert state_machine.state == States.starting

        # NOTE: we are already in Starting and entering it again
        with extra(state_machine):
            state = state_machine.get_state(States.starting)

            entry_action = mocker.stub(name='entry action')
            state.add_action(lambda: entry_action(), statesman.Action.Types.entry)
            exit_action = mocker.stub(name='exit action')
            state.add_action(lambda: exit_action(), statesman.Action.Types.exit)

            on_callback_mock = mocker.spy(state_machine, 'on_transition')
            await state_machine.enter_state(States.starting, 1234, foo='bar', type_=statesman.Transition.Types.internal)
            on_callback_mock.assert_called_once()

            entry_action.assert_not_called()
            exit_action.assert_not_called()

    async def test_runs_state_actions_for_self_transitions(self, mocker) -> None:
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States), state=States.starting)
        assert state_machine.state == States.starting

        # NOTE: we are already in Starting and entering it again
        with extra(state_machine):
            state = state_machine.get_state(States.starting)

            entry_action = mocker.stub(name='entry action')
            state.add_action(lambda: entry_action(), statesman.Action.Types.entry)
            exit_action = mocker.stub(name='exit action')
            state.add_action(lambda: exit_action(), statesman.Action.Types.exit)

            on_callback_mock = mocker.spy(state_machine, 'on_transition')
            await state_machine.enter_state(States.starting, 1234, foo='bar', type_=statesman.Transition.Types.self)
            on_callback_mock.assert_called_once()

            entry_action.assert_called_once()
            exit_action.assert_called_once()

    @pytest.mark.parametrize(('target_state',
                              'transition_type',
                              'error_message'),
                             [(States.starting,
                               statesman.Transition.Types.external,
                               'source and target states cannot be the same for external transitions'),
                              (States.running,
                                 statesman.Transition.Types.internal,
                                 'source and target states must be the same for internal or self transitions'),
                              (States.stopping,
                                 statesman.Transition.Types.self,
                                 'source and target states must be the same for internal or self transitions'),
                              ],
                             )
    async def test_raises_if_states_and_transition_type_are_inconsistent(
        self, target_state: statesman.StateEnum, transition_type: statesman.Transition.Types, error_message: str
    ) -> None:
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States), state=States.starting)
        assert state_machine.state == States.starting

        with pytest.raises(pydantic.ValidationError, match=error_message):
            await state_machine.enter_state(target_state, type_=transition_type)

    class TestEntryConfig:
        async def test_allow(self) -> None:
            state_machine = statesman.StateMachine(states=statesman.State.from_enum(States))
            state_machine.__config__.state_entry = statesman.Entry.allow
            assert state_machine.state is None
            assert await state_machine.enter_state(States.starting)
            assert state_machine.state == States.starting
            assert await state_machine.enter_state(States.stopping)
            assert state_machine.state == States.stopping
            assert await state_machine.enter_state(States.stopped)
            assert state_machine.state == States.stopped

        async def test_initial(self) -> None:
            # Enter once for initial, then raise on next try
            state_machine = statesman.StateMachine(states=statesman.State.from_enum(States))
            state_machine.__config__.state_entry = statesman.Entry.initial
            assert state_machine.state is None
            assert await state_machine.enter_state(States.starting)
            assert state_machine.state == States.starting

            with pytest.raises(RuntimeError, match="state entry failed: `enter_state` is only available to set initial state"):
                assert await state_machine.enter_state(States.stopping)

        async def test_ignore(self) -> None:
            # Return false every time
            state_machine = statesman.StateMachine(states=statesman.State.from_enum(States))
            state_machine.__config__.state_entry = statesman.Entry.ignore
            assert state_machine.state is None
            assert not await state_machine.enter_state(States.starting)
            assert state_machine.state is None
            assert not await state_machine.enter_state(States.stopping)
            assert state_machine.state is None
            assert not await state_machine.enter_state(States.stopped)
            assert state_machine.state is None

        async def test_forbid(self) -> None:
            state_machine = statesman.StateMachine(states=statesman.State.from_enum(States))
            state_machine.__config__.state_entry = statesman.Entry.forbid
            assert state_machine.state is None
            with pytest.raises(RuntimeError, match="state entry failed: use of the `enter_state` method is forbidden"):
                assert await state_machine.enter_state(States.starting)

    def test_add_event_fails_if_existing(self) -> None:
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States), state=States.starting)
        state = state_machine.states[0]
        state_machine.add_event(
            statesman.Event(
                name='finish',
                sources=[state],
                target=state,
            ),
        )
        with pytest.raises(ValueError, match='an event named "finish" already exists'):
            state_machine.add_event(
                statesman.Event(
                    name='finish',
                    sources=[state],
                    target=state,
                ),
            )

    def test_add_event_fails_with_unknown_state(self) -> None:
        state_machine = statesman.StateMachine()
        state = statesman.State('invalid')
        with pytest.raises(ValueError, match='cannot add an event that references unknown states: "invalid"'):
            state_machine.add_event(
                statesman.Event(
                    name='finish',
                    sources=[state],
                    target=state,
                ),
            )

    def test_add_event_allows_active_sentinel_state(self) -> None:
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States))
        state_machine.add_event(
            statesman.Event(
                name='finish',
                sources=state_machine.states,
                target=statesman.State.active(),
            ),
        )

    def test_cant_remove_active_state(self) -> None:
        state_machine = statesman.StateMachine()
        with pytest.raises(ValueError, match='cannot remove the active State'):
            state_machine.remove_state(statesman.State.active())

    def test_removing_state_clears_all_referencing_events(self) -> None:
        state_machine = statesman.StateMachine(states=statesman.State.from_enum(States), state=States.starting)
        state = state_machine.states[0]
        event = statesman.Event(
            name='finish',
            sources=[state],
            target=state,
        )
        state_machine.add_event(event)
        assert state_machine.events == [event]

        state_machine.remove_state(state)
        assert state_machine.events == []

    class TestTrigger:
        @pytest.fixture()
        def state_machine(self) -> statesman.StateMachine:
            state_machine = statesman.StateMachine()
            state_machine.add_states([
                statesman.State(
                    name=States.starting,
                ),
                statesman.State(
                    name=States.stopping,
                ),
            ])
            state_machine.add_event(
                statesman.Event(
                    name='finish',
                    sources=state_machine.get_states(States.starting, States.running),
                    target=state_machine.get_state(States.stopping),
                ),
            )
            state_machine.add_event(
                statesman.Event(
                    name='reset',
                    sources=state_machine.get_states(States.stopping),
                    target=state_machine.get_state(States.starting),
                ),
            )
            return state_machine

        async def test_get_event(self, state_machine: statesman.StateMachine) -> None:
            event = state_machine.get_event('finish')
            assert event is not None

        async def test_get_event_not_found(self, state_machine: statesman.StateMachine) -> None:
            event = state_machine.get_event('invalid')
            assert event is None

        async def test_get_event_invalid_type_raises(self, state_machine: statesman.StateMachine) -> None:
            assert state_machine.state is None
            with pytest.raises(TypeError) as e:
                state_machine.get_event(1234)

            assert str(e.value) == "cannot get event for name of type \"int\": 1234"

        async def test_can_trigger(self, state_machine: statesman.StateMachine) -> None:
            await state_machine.enter_state(States.starting)
            assert state_machine.state == States.starting
            assert state_machine.can_trigger_event('finish')
            assert not state_machine.can_trigger_event('reset')
            await state_machine.trigger_event('finish')
            assert state_machine.state == States.stopping
            assert not state_machine.can_trigger_event('finish')
            assert state_machine.can_trigger_event('reset')

        async def test_can_trigger_from_state(self, state_machine: statesman.StateMachine) -> None:
            assert state_machine.can_trigger_event('finish', from_state=States.starting)
            assert state_machine.can_trigger_event('finish', from_state="starting")
            assert state_machine.can_trigger_event('finish', from_state=state_machine.get_state("starting"))

            assert not state_machine.can_trigger_event('reset', from_state=States.starting)
            assert not state_machine.can_trigger_event('reset', from_state="starting")
            assert not state_machine.can_trigger_event('reset', from_state=state_machine.get_state("starting"))

        async def test_can_trigger_from_state(self, state_machine: statesman.StateMachine) -> None:
            assert state_machine.triggerable_events() == []
            assert state_machine.triggerable_events(from_state=None) == []
            assert state_machine.triggerable_events(from_state="stopping") == [state_machine.get_event("reset")]
            assert state_machine.triggerable_events(from_state="starting") == [state_machine.get_event("finish")]

        async def test_by_name(self, state_machine: statesman.StateMachine) -> None:
            await state_machine.enter_state(States.starting)
            assert state_machine.state == States.starting
            await state_machine.trigger_event('finish')
            assert state_machine.state == States.stopping

        async def test_by_event(self, state_machine: statesman.StateMachine) -> None:
            await state_machine.enter_state(States.starting)
            assert state_machine.state == States.starting
            event = state_machine.get_event('finish')
            await state_machine.trigger_event(event)
            assert state_machine.state == States.stopping

        async def test_trigger_without_state_raises(self, state_machine: statesman.StateMachine) -> None:
            assert state_machine.state is None
            with pytest.raises(RuntimeError, match='event trigger failed: the "finish" event does not support initial state transitions'):
                await state_machine.trigger_event('finish')

        async def test_trigger_from_incompatible_state(self, state_machine: statesman.StateMachine) -> None:
            await state_machine.enter_state(States.stopping)
            with pytest.raises(RuntimeError, match='event trigger failed: the "finish" event cannot be triggered from the current state of "stopping"'):
                await state_machine.trigger_event('finish')

        async def test_with_invalid_name(self, state_machine: statesman.StateMachine) -> None:
            await state_machine.enter_state(States.starting)
            with pytest.raises(LookupError, match="event trigger failed: no event was found with the name \"invalid\""):
                await state_machine.trigger_event('invalid')

        async def test_with_invalid_type(self, state_machine: statesman.StateMachine) -> None:
            await state_machine.enter_state(States.starting)
            with pytest.raises(TypeError, match="event trigger failed: cannot trigger an event of type \"int\": 1234"):
                await state_machine.trigger_event(1234)

        class TestReturnTypes:
            @pytest.fixture()
            def event(self, state_machine: statesman.StateMachine) -> statesman.Event:
                event = state_machine.get_event('finish')
                event.add_action(lambda: 31337, statesman.Action.Types.on)
                event.add_action(lambda: 187, statesman.Action.Types.on)
                event.add_action(lambda: 420, statesman.Action.Types.on)
                return event

            @pytest.mark.parametrize(('return_type', 'expected_return_value',),
                [
                    (bool, True),
                    (object, 31337),
                    (tuple, (True, 31337)),
                    (list, [31337, 187, 420]),
                ]
            )
            async def test_return_types(self, state_machine: statesman.StateMachine, event, return_type, expected_return_value) -> None:
                await state_machine.enter_state(States.starting)
                assert state_machine.state == States.starting
                event.return_type = return_type
                result = await state_machine.trigger_event('finish')
                assert result == expected_return_value

            @pytest.mark.parametrize(('return_type', 'expected_return_value',),
                [
                    (bool, True),
                    (object, 31337),
                    (tuple, (True, 31337)),
                    (list, [31337, 187, 420]),
                ]
            )
            async def test_return_types_override_on_trigger(self, state_machine: statesman.StateMachine, event, return_type, expected_return_value) -> None:
                await state_machine.enter_state(States.starting)
                assert state_machine.state == States.starting
                event.return_type = statesman.Transition
                result = await state_machine.trigger_event('finish', return_type=return_type)
                assert result == expected_return_value

            async def test_transition_return_type(self, state_machine: statesman.StateMachine, event) -> None:
                await state_machine.enter_state(States.starting)
                assert state_machine.state == States.starting
                event.return_type = statesman.Transition
                transition = await state_machine.trigger_event('finish')
                assert isinstance(transition, statesman.Transition)
                assert transition.event == event
                assert transition.succeeded == True
                assert transition.results == [31337, 187, 420]

        async def test_with_event_not_in_machine(self, state_machine: statesman.StateMachine) -> None:
            invalid_event = statesman.Event(
                name='invalid',
                sources=state_machine.states,
                target=state_machine.get_state(States.stopping),
            )
            await state_machine.enter_state(States.starting)
            with pytest.raises(TypeError, match="event trigger failed: cannot trigger an event of type \"int\": 1234"):
                await state_machine.trigger_event(1234)

        async def test_cancel_via_guard_state_machine_method(self, state_machine: statesman.StateMachine, mocker) -> None:
            await state_machine.enter_state(States.starting)
            with extra(state_machine):
                guard_mock = mocker.patch.object(state_machine, 'guard_transition')
                guard_mock.return_value = False
                success = await state_machine.trigger_event('finish')
                guard_mock.assert_awaited_once()
                assert not success, 'should have been guarded'

        async def test_returning_none_from_guard_does_not_cancel(self, state_machine: statesman.StateMachine, mocker) -> None:
            await state_machine.enter_state(States.starting)
            with extra(state_machine):
                guard_mock = mocker.patch.object(state_machine, 'guard_transition')
                guard_mock.return_value = None
                success = await state_machine.trigger_event('finish')
                guard_mock.assert_awaited_once()
                assert success, 'should not have been guarded'

        async def test_returning_invalid_value_from_guard_raises_value_error(self, state_machine: statesman.StateMachine, mocker) -> None:
            await state_machine.enter_state(States.starting)
            with extra(state_machine):
                guard_mock = mocker.patch.object(state_machine, 'guard_transition')
                guard_mock.return_value = "invalid"
                with pytest.raises(ValueError, match="invalid return value from guard_transition: must return True, False, or None"):
                    await state_machine.trigger_event('finish')
                guard_mock.assert_awaited_once()

        async def test_non_assertion_errors_raise(self, state_machine: statesman.StateMachine, mocker) -> None:
            await state_machine.enter_state(States.starting)
            with extra(state_machine):
                guard_mock = mocker.patch.object(state_machine, 'guard_transition')
                guard_mock.side_effect = RuntimeError(f'failed!')

                with pytest.raises(RuntimeError, match='failed!'):
                    success = await state_machine.trigger_event('finish')
                    guard_mock.assert_awaited_once()
                    assert not success, 'should have been guarded'

        async def test_guard_with_silence(self, state_machine: statesman.StateMachine, mocker) -> None:
            state_machine.__config__.guard_with = statesman.Guard.silence
            await state_machine.enter_state(States.starting)
            with extra(state_machine):
                guard_mock = mocker.patch.object(state_machine, 'guard_transition')
                guard_mock.return_value = False
                success = await state_machine.trigger_event('finish')
                guard_mock.assert_awaited_once()
                assert not success, 'should have been guarded'

        async def test_guard_with_warning(self, state_machine: statesman.StateMachine, mocker) -> None:
            state_machine.__config__.guard_with = statesman.Guard.warning
            await state_machine.enter_state(States.starting)
            with extra(state_machine):
                guard_mock = mocker.patch.object(state_machine, 'guard_transition')
                guard_mock.return_value = False
                with pytest.warns(UserWarning, match='transition guard failure: guard_transition returned False'):
                    await state_machine.trigger_event('finish')

        async def test_guard_with_exception(self, state_machine: statesman.StateMachine, mocker) -> None:
            state_machine.__config__.guard_with = statesman.Guard.exception
            await state_machine.enter_state(States.starting)
            with extra(state_machine):
                guard_mock = mocker.patch.object(state_machine, 'guard_transition')
                guard_mock.return_value = False
                with pytest.raises(RuntimeError, match="transition guard failure: guard_transition returned False"):
                    await state_machine.trigger_event('finish')

        class TestActions:
            async def test_guard(self, state_machine: statesman.StateMachine, mocker) -> None:
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event('finish')
                guard_action = mocker.stub(name='action')
                guard_action.return_value = True
                event.add_action(lambda: guard_action(), statesman.Action.Types.guard)
                assert await state_machine.trigger_event('finish'), 'guard passed'
                guard_action.assert_called_once()

            async def test_cancel_via_guard_action_bool(self, state_machine: statesman.StateMachine, mocker) -> None:
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event('finish')
                guard_action = mocker.stub(name='action')
                guard_action.return_value = False

                event.add_action(lambda: guard_action(), statesman.Action.Types.guard)
                # NOTE: The AssertionError is being caught and aborts the test
                success = await state_machine.trigger_event('finish')
                guard_action.assert_called_once()
                assert not success, 'should have been cancelled by guard'

            async def test_none_return_value_from_guard_does_not_cancel(self, state_machine: statesman.StateMachine, mocker) -> None:
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event('finish')
                guard_action = mocker.stub(name='action')
                guard_action.return_value = None

                event.add_action(lambda: guard_action(), statesman.Action.Types.guard)
                success = await state_machine.trigger_event('finish')
                guard_action.assert_called_once()
                assert success, 'should not have been cancelled by guard'

            async def test_invalid_return_value_from_guard_raises_value_error(self, state_machine: statesman.StateMachine, mocker) -> None:
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event('finish')
                guard_action = mocker.stub(name='action')
                guard_action.return_value = "invalid"

                event.add_action(lambda: guard_action(), statesman.Action.Types.guard)
                with pytest.raises(ValueError, match="invalid return value from guard action: must return True, False, or None"):
                    await state_machine.trigger_event('finish')
                guard_action.assert_called_once()

            async def test_cancel_via_guard_action_exception(self, state_machine: statesman.StateMachine, mocker) -> None:
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event('finish')
                guard_action = mocker.stub(name='action')
                guard_action.side_effect = AssertionError('should be suppressed')

                event.add_action(lambda: guard_action(), statesman.Action.Types.guard)
                # NOTE: The AssertionError is being caught and aborts the test
                success = await state_machine.trigger_event('finish')
                assert not success, 'cancelled by guard'
                guard_action.assert_called_once()

            async def test_guard_actions_run_sequentially(self, state_machine: statesman.StateMachine, mocker) -> None:
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event('finish')
                guard_action1 = mocker.stub(name='first guard')
                guard_action1.return_value = False
                guard_action2 = mocker.stub(name='second guard')
                guard_action2.return_value = True
                event.add_action(lambda: guard_action1(), statesman.Action.Types.guard)
                event.add_action(lambda: guard_action2(), statesman.Action.Types.guard)
                assert not await state_machine.trigger_event('finish'), 'guard failed'
                guard_action1.assert_called_once()
                guard_action2.assert_not_called()

            async def test_guard_with_silence(self, state_machine: statesman.StateMachine, mocker) -> None:
                state_machine.__config__.guard_with = statesman.Guard.silence
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event('finish')
                guard_action = mocker.stub(name='action')
                guard_action.return_value = False

                event.add_action(lambda: guard_action(), statesman.Action.Types.guard)
                success = await state_machine.trigger_event('finish')

            async def test_guard_with_warning(self, state_machine: statesman.StateMachine, mocker) -> None:
                state_machine.__config__.guard_with = statesman.Guard.warning
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event('finish')
                guard_action = mocker.stub(name='action')
                guard_action.return_value = False

                event.add_action(lambda: guard_action(), statesman.Action.Types.guard)
                with pytest.warns(UserWarning, match='transition guard failure: guard action returned False'):
                    await state_machine.trigger_event('finish')

            async def test_guard_with_exception(self, state_machine: statesman.StateMachine, mocker) -> None:
                state_machine.__config__.guard_with = statesman.Guard.exception
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event('finish')
                guard_action = mocker.stub(name='action')
                guard_action.return_value = False

                event.add_action(lambda: guard_action(), statesman.Action.Types.guard)
                with pytest.raises(RuntimeError, match="transition guard failure: guard action returned False"):
                    await state_machine.trigger_event('finish')

            async def test_before(self, state_machine: statesman.StateMachine, mocker) -> None:
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event('finish')
                before_action = mocker.stub(name='action')
                event.add_action(lambda: before_action(), statesman.Action.Types.before)
                await state_machine.trigger_event('finish')
                before_action.assert_called_once()

            async def test_after(self, state_machine: statesman.StateMachine, mocker) -> None:
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event('finish')
                after_action = mocker.stub(name='action')
                event.add_action(lambda: after_action(), statesman.Action.Types.after)
                await state_machine.trigger_event('finish')
                after_action.assert_called_once()

            async def test_on(self, state_machine: statesman.StateMachine, mocker) -> None:
                await state_machine.enter_state(States.starting)
                event = state_machine.get_event('finish')
                on_action = mocker.stub(name='action')
                event.add_action(lambda: on_action(), statesman.Action.Types.on)
                await state_machine.trigger_event('finish')
                on_action.assert_called_once()

            async def test_inheritable_actions(self, state_machine: statesman.StateMachine, mocker) -> None:
                with extra(state_machine):
                    guard_transition = mocker.spy(state_machine, 'guard_transition')
                    before_transition = mocker.spy(state_machine, 'before_transition')
                    on_transition = mocker.spy(state_machine, 'on_transition')
                    after_transition = mocker.spy(state_machine, 'after_transition')
                    await state_machine.enter_state(States.starting)
                    await state_machine.trigger_event('finish')

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
        starting = 'Starting...'
        running = 'Running...'
        stopping = 'Stopping...'
        stopped = 'Terminated.'

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
                one = 'One'
                two = 'Two'
                three = 'Three'
                four = 'Four'

        state_machine = TestMachine()
        assert state_machine.state is None
        assert len(state_machine.states) == 4
        assert state_machine.states[0] == TestMachine.States.one
        assert state_machine.state is None

    class ProcessLifecycle(statesman.StateMachine):
        class States(statesman.StateEnum):
            starting = 'Starting...'
            running = 'Running...'
            stopping = 'Stopping...'
            stopped = 'Terminated.'

        # Track state about the process we are running
        command: Optional[str] = None
        pid: Optional[int] = None
        logs: List[str] = []

        # initial state entry point
        @statesman.event(None, States.starting)
        async def start(self, command: str = '...') -> None:
            """Start a process."""
            self.command = command
            self.pid = 31337
            self.logs.clear()  # Flush logs between runs

        @statesman.event(source=States.starting, target=States.running)
        async def run(self, transition: statesman.Transition) -> None:
            """Mark as process as running."""
            self.logs.append(f'Process pid {self.pid} is now running (command=\"{self.command}\")')

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

        @statesman.event(source=States.stopped, target=States.stopped,
                         transition_type=statesman.Transition.Types.self)
        async def halt(self) -> None:
            """Halt the process."""
            self.logs.append(f'Halted')

        def _is_okay(self) -> bool:
            return True

        @statesman.event(States.__any__, States.__active__, guard=_is_okay)
        async def _reset(self) -> None:
            """Reset the state machine."""
            ...

        @statesman.enter_state(States.stopped)
        async def _on_stop(self) -> None:
            self.logs.append(f'_on_stop triggered')

        @statesman.enter_state([States.starting, States.running])
        async def _enter_states(self, state) -> None:
            self.logs.append(f'_enter_states triggered: {state}')

        @statesman.after_event('terminate')
        async def _terminated(self) -> None:
            self.logs.append(f'_terminated')

        async def after_transition(self, transition: statesman.Transition) -> None:
            if transition.event and transition.event.name == 'stop':
                await self.terminate()

    @pytest.fixture()
    def state_machine(self) -> statesman.StateMachine:
        return TestDecoratorStateMachine.ProcessLifecycle()

    async def test_states_added(self, state_machine: statesman.StateMachine) -> None:
        assert len(state_machine.states) == 4
        assert state_machine.states[0].name == 'starting'
        assert state_machine.states[0].description == 'Starting...'

    async def test_events_added(self, state_machine: statesman.StateMachine) -> None:
        event = state_machine.get_event('start')
        assert event
        assert event.description == 'Start a process.'
        assert event.sources == [None]
        assert event.target == States.starting

    async def test_trigger_event_through_method_call(self, state_machine: statesman.StateMachine) -> None:
        assert state_machine.state is None
        await state_machine.start()
        assert state_machine.state
        assert state_machine.state == States.starting

    async def test_trigger_event_through_method_call_with_args(self, state_machine: statesman.StateMachine, mocker) -> None:
        with extra(state_machine):
            callback_mock = mocker.spy(state_machine, 'on_transition')
            await state_machine.start(31337, this='That')
            callback_mock.assert_called_once()
            assert 31337 in callback_mock.call_args.args
            assert {'this': 'That'} == callback_mock.call_args.kwargs

    async def test_self_transition(self, state_machine: statesman.StateMachine, mocker) -> None:
        await state_machine.enter_state(States.stopped)

        state = state_machine.get_state(States.stopped)
        assert state_machine.state == state

        event = state_machine.get_event('halt')
        assert event.transition_type == statesman.Transition.Types.self

        with extra(state_machine):
            entry_stub = mocker.stub(name='entering halting')
            exit_stub = mocker.stub(name='exiting halting')
            def entry(): return entry_stub()
            def exit(): return exit_stub()
            state.add_action(entry, statesman.Action.Types.entry)
            state.add_action(exit, statesman.Action.Types.exit)

            await state_machine.halt()
            entry_stub.assert_called_once()
            exit_stub.assert_called_once()

    async def test_process_lifecycle(self, state_machine: statesman.StateMachine, mocker) -> None:
        assert state_machine.pid is None
        assert state_machine.command is None

        await state_machine.start('ls -al')
        assert state_machine.command == 'ls -al'
        assert state_machine.pid == 31337
        assert state_machine.state == States.starting
        assert len(state_machine.logs) == 1, f"Got unexpected logs: {state_machine.logs}"

        await state_machine.run()
        assert 'Process pid 31337 is now running (command="ls -al")' in state_machine.logs

        await state_machine.stop()
        assert state_machine.logs == [
            "_enter_states triggered: name='starting' description='Starting...'",
            'Process pid 31337 is now running (command="ls -al")',
            "_enter_states triggered: name='running' description='Running...'",
            'Shutting down pid 31337 (command="ls -al")',
            'Terminated pid 31337 ("ls -al")',
            '_on_stop triggered',
            '_terminated',
        ]

        # Let the runloop cycle
        await asyncio.sleep(0.001)
        assert state_machine.state == States.stopped
        assert state_machine.pid is None
        assert state_machine.command is None

    async def test_enter_states(self, state_machine: statesman.StateMachine) -> None:
        assert state_machine.state is None
        await state_machine.enter_state(States.starting)
        assert state_machine.state
        assert state_machine.state == States.starting
        await state_machine.enter_state(States.running)
        assert state_machine.logs == [
            "_enter_states triggered: name='starting' description='Starting...'",
            "_enter_states triggered: name='running' description='Running...'",
        ]


class TestInitialState:
    class StateMachine(statesman.StateMachine):
        class States(statesman.StateEnum):
            starting = 'Starting...'
            running = 'Running...'
            stopping = 'Stopping...'
            stopped = statesman.InitialState('Terminated.')

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
                one = statesman.InitialState('1')
                two = '2'
                three = '3'
                four = statesman.InitialState('4')

@pytest.fixture(autouse=True)
async def reset_config() -> None:
    state_entry = statesman.StateMachine.__config__.state_entry
    guard_with = statesman.StateMachine.__config__.guard_with
    try:
        yield
    finally:
        statesman.StateMachine.__config__.state_entry = state_entry
        statesman.StateMachine.__config__.guard_with = guard_with


@contextlib.contextmanager
def extra(
    obj: pydantic.BaseModel, extra: pydantic.Extra = pydantic.Extra.allow,
) -> Iterator[pydantic.BaseModel]:
    """Temporarily override the value of the `extra` setting on a Pydantic
    object.

    Used in tests to support object mocking/spying that relies on
    setattr to inject mocks.
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

    args = ('whatever',)
    kwargs = {'transition': 'foo'}

    await statesman._call_with_matching_parameters(some_function, *args, **kwargs)

class TestSequencer:
    class StateMachine(statesman.SequencingMixin, statesman.StateMachine):
        class States(statesman.StateEnum):
            starting = 'Starting...'
            running = 'Running...'
            stopping = 'Stopping...'
            stopped = statesman.InitialState('Terminated.')

        name: Optional[str] = None
        count: Optional[int] = None
        another: Optional[str] = None
        reason: Optional[str] = None

        # _queue: asyncio.Queue = asyncio.Queue()

        @statesman.event(States.stopped, States.starting)
        async def start(self, name: str) -> None:
            ...

        @statesman.event(States.starting, States.running)
        async def run(self, *, count: int, another: str) -> None:
            ...

        @statesman.event(States.running, States.stopping)
        async def stop(self, *, reason: str = "Foo") -> None:
            ...

        @statesman.event(States.stopping, States.stopped)
        async def terminate(self, *args, **kwargs) -> None:
            ...

    async def test_sequencing(self) -> None:
        state_machine = await TestSequencer.StateMachine.create()
        state_machine.sequence(
            state_machine.start(name="Foo"),
            state_machine.run(count=5, another="whatever"),
            state_machine.trigger_event("stop"),
            state_machine.terminate(),
            state_machine.enter_state(States.starting),
        )

        expected_states = [States.starting, States.running, States.stopping, States.stopped, States.starting]
        for expected_state in expected_states:
            transition = await state_machine.next_state()
            assert isinstance(transition, statesman.Transition)
            assert transition.succeeded
            assert state_machine.state == expected_state

        result = await state_machine.next_state()
        assert result is None
