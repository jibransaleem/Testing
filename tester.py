
from a2a_caller.a2a_client_abstraction import KnowledgeAgent


PROJECT_ID = "755c5457-12e1-5dbe-8997-cfb6990ac508"


def test_get_context():
    """Test retrieving relevant project context."""

    with KnowledgeAgent() as kb:
        response = kb.get_context(
            project_id=PROJECT_ID,
            query="What are the functional requirements?"
        )

        print("\nGET CONTEXT RESPONSE:")
        print(response)

        assert response is not None
        assert isinstance(response, dict)


def test_get_package():
    """Test retrieving the project knowledge package."""

    with KnowledgeAgent() as kb:
        response = kb.get_package(
            project_id=PROJECT_ID,
            package_type="UC1_CONTEXT"
        )

        print("\nGET PACKAGE RESPONSE:")
        print(response)

        assert response is not None
        assert isinstance(response, dict)


if __name__ == "__main__":
    test_get_context()
    test_get_package()

    print("\nBoth tests passed!")
