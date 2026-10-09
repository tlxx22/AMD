"""Only the two authorized width candidates; no third-round continuation."""
from utils.ch3_patchtst_width_reproduction import activate
activate()
from m6_type1_followup_entry import cli

if __name__=='__main__':
    import sys
    sys.exit(cli())
