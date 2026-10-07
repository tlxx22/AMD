"""Fixed attempt selection; public CLI, guards and chain remain shared in N."""
from utils.ch3_probe_schema_recovery import activate
activate()
from m6_type1_followup_entry import cli
if __name__=='__main__':
    import sys
    sys.exit(cli())
