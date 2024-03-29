# Ref: https://github.com/scipy/scipy/blob/main/.cirrus.star

# The guide to programming cirrus-ci tasks using starlark is found at
# https://cirrus-ci.org/guide/programming-tasks/
#
# In this simple starlark script we simply check conditions for whether
# a CI run should go ahead. If the conditions are met, then we just
# return the yaml containing the tasks to be run.

load("cirrus", "re", "env", "fs", "http")

BUILD_PATTERN = re.compile(
    r"\[wheel build(: (?P<build_ref>.+?))?\]"
)

# This may match false postives but it is good enough.
# Bad matches will be caught in the python check
SEMVER_TAG_PATTERN = re.compile(
    r"^v\d+\.\d+\.\d+((a|b|rc)\d+)?$"
)

def main(ctx):
    ######################################################################
    # Should wheels be built?
    # Only test on the scipy/scipy repository
    # Test if the run was triggered by:
    # - a cron job called "nightly". The cron job is not set in this file,
    #   but on the cirrus-ci repo page
    # - commit message containing [wheel build]
    ######################################################################
    if env.get("CIRRUS_REPO_FULL_NAME") != "has2k1/scikit-misc":
        return []

    # Obtain commit message for the event. Unfortunately CIRRUS_CHANGE_MESSAGE
    # only contains the actual commit message on a non-PR trigger event.
    # For a PR event it contains the PR title and description.
    SHA = env.get("CIRRUS_CHANGE_IN_REPO")
    url = "https://api.github.com/repos/has2k1/scikit-misc/git/commits/" + SHA
    dct = http.get(url).json()
    message = dct["message"]
    tag = env.get("CIRRUS_TAG")

    # this configuration runs a single linux_aarch64 + macosx_arm64 run.
    # there's no need to do this during a wheel run as they automatically build
    # and test over a wider range of Pythons.
    if "[skip cirrus]" in message or "[skip ci]" in message:
        return []

    m = BUILD_PATTERN.search(message)
    m2 = SEMVER_TAG_PATTERN.match(tag) if tag else None
    if not m and not m2:
        return []
    return fs.read("ci/cirrus_wheels.yml")
