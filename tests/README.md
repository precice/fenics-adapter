# Tests

This folder contains everything needed for testing. The tests are split into two categories:

* `unit` contains unit tests that only check independent functions and modules. Interaction with preCICE is not required. Therefore no mocking should be performed.
* `integration` contains integration tests that interact with preCICE. For this purpose mocking is used extensively.

## Programming Guidelines

Make sure to only use `@patch.dict('sys.modules', **{'precice': tests.MockedPrecice})` in `integration` and not in `unit`. If during the development of a test mocking becomes necessary or the mocked up version of preCICE is not used you might have to reconsider the design or where the test is located.
