# CHANGELOG

statesman is a modern Python library for elegantly implementing state machines.

statesman is distributed under the terms of the Apache 2.0 license.

This changelog catalogs all notable changes made to the project. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Releases are
versioned in accordance with [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] "crimson meerkat" - 11/23/2021

### Changed

- Default python version set to 3.9.9
- (post1) Relax python version requirements to allow installation on versions as low as 3.7

### Fixed

- Bug where pydantic model Event.return_type default value (bool) was overriding initialization argument

## [1.0.1] "crimson meerkat" - 11/27/2020

Patch release to expand package discovery metadata. No functional changes.

## [1.0.0] "crimson meerkat" - 11/27/2020

Initial stable release of Statesman. Hear me and rejoice.

### Added

* State machines have optional history tracking via the `statesman.HistoryMixin`
    module.
* State machine transitions can be explicitly sequenced via the `statesman.SequencingMixin`
    module.
* The functionality of the `enter_state` method can now be constrained via the
    nested config class. This enables state machines to be restricted to
    operating purely through transitions.
* Guard behaviors can be configured via the `guard_with` option. This enables
    guard actions to operate silently, exceptionally, or emit warning log
    messages.
* The `triggerable_events` method can be used to retrieve a list of events that
    can be triggered from a given state.
* Decorators now utilize the formatted docstring of the function they are
    wrapping to set the description of the state/event/action.
* State machine objects now support inheritance.
* The `enter_state` and `exit_state` decorators now support attaching a handler
    to multiple states at once.

### Changed

* The `trigger` method has been renamed to `trigger_event`.
* The `can_trigger` method has been renamed to `can_trigger_event`.
* Description arguments have been removed from the positional argument list of
    decorators.
* Guard actions are now run sequentially in registration order rather than
    being gathered asynchronously.

## [0.1.0]

Initial public release.
