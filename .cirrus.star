# Ref: https://github.com/scipy/scipy/blob/main/.cirrus.star

# The guide to programming cirrus-ci tasks using starlark is found at
# https://cirrus-ci.org/guide/programming-tasks/
#
# In this simple starlark script we simply check conditions for whether
# a CI run should go ahead. If the conditions are met, then we just
# return the yaml containing the tasks to be run.

load("cirrus", "re", "env", "fs", "http")

BUILD_TAG_PATTERN = re.compile(
    r"\[wheel build(: (?P<build_ref>.+?))?\]$"
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

    # this configuration runs a single linux_aarch64 + macosx_arm64 run.
    # there's no need to do this during a wheel run as they automatically build
    # and test over a wider range of Pythons.
    if "[skip cirrus]" in message or "[skip ci]" in message:
        return []

    c1 = "[wheel build]" in message
    c2  = "[wheel build - cirrus]" in message
    c3 = BUILD_TAG_PATTERN.match(message) != None
    if c1 or c2 or c3:
        return fs.read("ci/cirrus_wheels.yml")

    return []


