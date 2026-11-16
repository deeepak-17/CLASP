"""Write the users file of a page: who may log in, with which role (U5, C2).

    python scripts/make_users.py --out .runtime/users/panel.json --add operator:operator
    python scripts/make_users.py --out .runtime/users/client-flask.json \\
        --add asha:developer:client-flask

``--add`` is ``name:role`` for the demo panel's ``operator``, or
``name:developer:<client id>`` for the developer of one edge page. The password
is asked for at the prompt, twice; it is never an argument, so it does not end
up in the shell history or the process list. The file holds a scrypt hash per
user and no password (``security.auth.users.UserStore``). Adding a name that
is already there replaces its password.

The panel reads its file from ``CLASP_USERS_FILE``; an edge page from
``--users``. ``scripts/make_edge_bundle.py`` puts
``.runtime/users/<client id>.json`` into that edge's bundle. A page with no
users file runs without a login and says so when it starts.
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USERS_DIR = REPO_ROOT / ".runtime" / "users"


def ask_password(prompt: str) -> str:
    return getpass.getpass(prompt)


def parse_spec(spec: str) -> tuple[str, str, str | None]:
    """``name:operator`` or ``name:developer:<client id>`` -> (name, role, subject)."""
    parts = spec.split(":")
    if len(parts) == 2 and parts[0] and parts[1] == "operator":
        return parts[0], "operator", None
    if len(parts) == 3 and parts[0] and parts[1] == "developer" and parts[2]:
        return parts[0], "developer", parts[2]
    raise argparse.ArgumentTypeError(
        f"{spec!r}: use name:operator, or name:developer:<client id> (the password is asked "
        f"for, never given here)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="write a page's users file (passwords are prompted)")
    ap.add_argument("--out", type=Path, required=True,
                    help="the users file, e.g. .runtime/users/panel.json")
    ap.add_argument("--add", type=parse_spec, action="append", required=True, metavar="SPEC",
                    help="name:operator, or name:developer:<client id>; repeat for more users")
    return ap


def main(argv=None) -> int:
    from security.auth.users import MIN_PASSWORD_LENGTH, UserStore, UserStoreError

    args = build_parser().parse_args(argv)
    try:
        store = UserStore(args.out)
    except UserStoreError as exc:
        print(f"make_users: {exc}", file=sys.stderr)
        return 1
    # ask for every password first, so a mistake writes nothing at all
    entries = []
    for name, role, subject in args.add:
        who = f"{name} ({role}{' of ' + subject if subject else ''})"
        password = ask_password(f"Password for {who}: ")
        if ask_password(f"Again for {who}: ") != password:
            print(f"make_users: the two passwords for {name} did not match; nothing was written",
                  file=sys.stderr)
            return 1
        if len(password) < MIN_PASSWORD_LENGTH:
            print(f"make_users: the password for {name} must be at least "
                  f"{MIN_PASSWORD_LENGTH} characters; nothing was written", file=sys.stderr)
            return 1
        entries.append((name, password, role, subject))
    for name, password, role, subject in entries:
        store.add(name, password, role, subject=subject)
    print(f"{args.out}: {', '.join(store.usernames())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
