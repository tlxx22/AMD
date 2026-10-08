"""Exactly one protected PatchTST enc1/ETTh1/H96 six-step serial acceptance."""
from utils.ch3_patchtst_depth_urban6_recovery import activate_serial_check
activate_serial_check()
from m6_type1_followup_entry import cli

if __name__ == '__main__':
    import sys
    sys.exit(cli())
