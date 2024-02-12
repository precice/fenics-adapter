# Guide to release new version of the FEniCS-preCICE adapter

Before starting this process make sure to check that all relevant changes are included in the `CHANGELOG.md`. The developer who is releasing a new version of FEniCS-preCICE adapter is expected to follow this workflow:

1. If it does not already exist, create a release branch with the version number of the planned release. Use develop as base for the branch. `git checkout develop`; `git checkout -b fenics-adapter-vX.X.X`. Perform the following steps only on the release branch, if not indicated differently.

2. [Open a Pull Request from the branch `fenics-adapter-vX.X.X` to `master`](https://github.com/precice/fenics-adapter/compare) named after the version (i.e. `Release v1.0.0`) and briefly describe the new features of the release in the PR description.

3. Bump the version in the following places:

    a) Before merging the PR, make sure to bump the version in `CHANGELOG.md` on `fenics-adapter-vX.X.X`.

    b) Update the version in `CITATION.cff` and update the release date.

    c) There is no need to bump the version anywhere else, since we use the [python-versioneer](https://github.com/python-versioneer/python-versioneer/) for maintaining the version everywhere else.

4. [Draft a New Release](https://github.com/precice/fenics-adapter/releases/new) in the `Releases` section of the repository page in a web browser. The release tag needs to be the exact version number (i.e.`v1.0.0` or `v1.0.0rc1`, compare to [existing tags](https://github.com/precice/fenics-adapter/tags)). Use `@target:master`. Release title is also the version number (i.e. `v1.0.0` or `v1.0.0rc1`, compare to [existing releases](https://github.com/precice/fenics-adapter/tags)).
*Note:* If it is a pre-release then the option *This is a pre-release* needs to be selected at the bottom of the page. Use `@target:fenics-adapter-vX.X.X` for a pre-release, since we will never merge a pre-release into master.

    a) If a pre-release is made: Directly hit the "Publish release" button in your Release Draft. Now you can check the artifacts (e.g. release on [PyPI](https://pypi.org/project/fenicsprecice/#history)) of the release. *Note:* As soon as a new tag is created github actions will take care of deploying the new version on PyPI using [this workflow](https://github.com/precice/fenics-adapter/actions?query=workflow%3A%22Upload+Python+Package%22).

    b) If this is a "real" release: As soon as one approving review is made, merge the release PR (`fenics-adapter-vX.X.X`) into `master`.

5. Merge `master` into `develop` for synchronization of `develop`.

6. If everything is in order up to this point then the new version can be released by hitting the "Publish release" button in your Release Draft.

7. Now there should be a tag for the release. Re-run the [docker release workflow `build-docker.yml` via dispatch](https://github.com/precice/fenics-adapter/actions/workflows/build-docker.yml) such that the correct version is picked up by `versioneer`. Check the version in the container via `docker pull precice/fenics-adapter`, then `docker run -ti precice/fenics-adapter`, and inside the container `$ python3 -c "import fenicsprecice; print(fenicsprecice.__version__)"`.
