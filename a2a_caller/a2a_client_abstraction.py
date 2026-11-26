

from .a2a_client import get_kb_client


class KnowledgeAgent:

    def __init__(self, timeout=60):
        """
        Initialize the A2A connection once.
        """

        self.client = get_kb_client(timeout=timeout)

    # -----------------------------------------------------
    # Get Context
    # -----------------------------------------------------

    def get_context(self, project_id: str, query: str):
        """
        Retrieve relevant context for a specific project.

        Args:
            project_id: Project UUID.
            query: Natural language question.

        Returns:
            dict: Knowledge Agent response.
        """

        if not project_id or not project_id.strip():
            raise ValueError("Project ID cannot be empty.")

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        request = {
            "project_id": project_id.strip(),
            "query": query.strip(),
        }

        return self.client.get_context(request)

    # -----------------------------------------------------
    # Get Package
    # -----------------------------------------------------

    def get_package(
        self,
        project_id: str,
        package_type: str = "UC1_CONTEXT",
    ):
        """
        Retrieve a project knowledge package.
        """

        if not project_id or not project_id.strip():
            raise ValueError("Project ID cannot be empty.")

        if not package_type or not package_type.strip():
            raise ValueError("Package type cannot be empty.")

        request = {
            "project_id": project_id.strip(),
            "package_type": package_type.strip(),
        }

        return self.client.get_package(request)

    # -----------------------------------------------------
    # Ping
    # -----------------------------------------------------

    def ping(self):
        return self.client.ping()

    # -----------------------------------------------------
    # Health
    # -----------------------------------------------------

    def health(self, deep=False):
        return self.client.health(deep=deep)

    # -----------------------------------------------------
    # Close
    # -----------------------------------------------------

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
