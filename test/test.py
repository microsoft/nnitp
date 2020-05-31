#
# Copyright (c) Microsoft Corporation.
#

# Run `nnitp` as a script.  If you try to run `nnitp.py` directy as a
# script, it fails because the Python module search path includes the
# source directly. This script allows you to run nnitp under pdb,
# like this:
#
#     $ pdb test.py args ...
#

import nnitp.nnitp
nnitp.nnitp.main()
