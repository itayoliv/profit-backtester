"""Agent-driven backtesting harness."""

__version__ = "0.1.0"

import truststore

# requests trusts only the certifi bundle. On this machine AVG HTTPS scanning
# presents its own root, which Windows already trusts and certifi does not.
truststore.inject_into_ssl()
