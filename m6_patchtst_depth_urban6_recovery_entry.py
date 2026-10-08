"""Adopt round2-371, execute depth48 and the expanded third-round287."""
from utils.ch3_patchtst_depth_urban6_recovery import activate
activate()
from m6_type1_followup_entry import cli

if __name__ == '__main__':
    import sys
    sys.exit(cli())
