"""Adopt the completed base287, then execute only the original remaining343."""
from utils.ch3_amend_ett_identity_recovery import activate
activate()
from m6_type1_followup_entry import cli

if __name__ == '__main__':
    import sys
    sys.exit(cli())
