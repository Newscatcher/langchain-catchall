"""Tests for CatchAllClient."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_catchall.client import CatchAllClient, PullJobResponseDto, Record


@pytest.fixture
def mock_client():
    """Create a client with mocked SDK."""
    return CatchAllClient(api_key="test_key")


def test_client_initialization():
    """Test client initializes with correct parameters."""
    client = CatchAllClient(
        api_key="test_key",
        base_url="https://test.api.com",
        poll_interval=60,
        max_wait_time=3600,
    )
    assert client.api_key == "test_key"
    assert client.poll_interval == 60
    assert client.max_wait_time == 3600
    assert client._client is not None


@patch("langchain_catchall.client.CatchAllApi")
def test_submit_job(mock_api_class):
    """Test job submission."""
    # Setup mock
    mock_api_instance = Mock()
    mock_api_class.return_value = mock_api_instance
    
    mock_response = Mock()
    mock_response.job_id = "test-job-123"
    mock_api_instance.jobs.create_job.return_value = mock_response
    
    # Test
    client = CatchAllClient(api_key="test_key")
    job_id = client.submit_job(query="test query", context="test context")
    
    assert job_id == "test-job-123"
    mock_api_instance.jobs.create_job.assert_called_once_with(
        query="test query",
        context="test context",
        schema=None,
    )


@patch("langchain_catchall.client.CatchAllApi")
def test_get_status(mock_api_class):
    """Test status checking."""
    # Setup mock
    mock_api_instance = Mock()
    mock_api_class.return_value = mock_api_instance
    
    mock_status = Mock()
    mock_status.job_id = "test-job-123"
    mock_status.status = "data_fetched"
    mock_api_instance.jobs.get_job_status.return_value = mock_status
    
    # Test
    client = CatchAllClient(api_key="test_key")
    status = client.get_status("test-job-123")
    
    assert status.status == "data_fetched"
    mock_api_instance.jobs.get_job_status.assert_called_once_with("test-job-123")


@patch("langchain_catchall.client.CatchAllApi")
@patch("langchain_catchall.client.time.sleep")
def test_wait_for_completion(mock_sleep, mock_api_class):
    """Test waiting for job completion."""
    # Setup mock
    mock_api_instance = Mock()
    mock_api_class.return_value = mock_api_instance
    
    # First call returns pending, second returns completed
    mock_status1 = Mock()
    mock_status1.status = "pending"
    
    mock_status2 = Mock()
    mock_status2.status = "job_completed"
    
    mock_api_instance.jobs.get_job_status.side_effect = [mock_status1, mock_status2]
    
    # Test
    client = CatchAllClient(api_key="test_key")
    client.wait_for_completion("test-job-123")
    
    assert mock_api_instance.jobs.get_job_status.call_count == 2
    mock_sleep.assert_called()


@patch("langchain_catchall.client.CatchAllApi")
@patch("langchain_catchall.client.time.time")
def test_wait_for_completion_timeout(mock_time, mock_api_class):
    """Test timeout when job takes too long."""
    # Setup mock
    mock_api_instance = Mock()
    mock_api_class.return_value = mock_api_instance
    
    # Simulate time passing
    mock_time.side_effect = [0, 0, 1300, 1300]  # Exceeds max_wait_time of 1200
    
    mock_status = Mock()
    mock_status.status = "pending"
    mock_api_instance.jobs.get_job_status.return_value = mock_status
    
    # Test
    client = CatchAllClient(api_key="test_key", max_wait_time=1200)
    
    with pytest.raises(TimeoutError):
        client.wait_for_completion("test-job-123")


@patch("langchain_catchall.client.CatchAllApi")
def test_get_results(mock_api_class):
    """Test retrieving results."""
    # Setup mock
    mock_api_instance = Mock()
    mock_api_class.return_value = mock_api_instance
    
    mock_result = PullJobResponseDto(
        job_id="test-job-123",
        status="job_completed",
        page=1,
        total_pages=1,
        page_size=100,
        valid_records=2,
        all_records=[
            Record(
                record_id="1",
                record_title="Test Record 1",
                enrichment={"key": "value"},
                citations=[{"title": "Source 1", "link": "http://example.com", "published_date": "2024-01-01 00:00:00"}]
            ),
            Record(
                record_id="2",
                record_title="Test Record 2",
                enrichment={"key": "value2"},
                citations=[]
            )
        ]
    )
    mock_api_instance.jobs.get_job_results.return_value = mock_result
    
    # Test
    client = CatchAllClient(api_key="test_key")
    result = client.get_results("test-job-123")
    
    assert isinstance(result, PullJobResponseDto)
    assert result.job_id == "test-job-123"
    assert len(result.all_records) == 2
    assert result.all_records[0].record_title == "Test Record 1"


@patch("langchain_catchall.client.CatchAllApi")
def test_search_with_wait(mock_api_class):
    """Test search method with waiting."""
    # Setup mock
    mock_api_instance = Mock()
    mock_api_class.return_value = mock_api_instance
    
    # Mock submit
    mock_submit_response = Mock()
    mock_submit_response.job_id = "test-job-123"
    mock_api_instance.jobs.create_job.return_value = mock_submit_response
    
    # Mock status
    mock_status = Mock()
    mock_status.status = "job_completed"
    mock_api_instance.jobs.get_job_status.return_value = mock_status
    
    # Mock results
    mock_result = PullJobResponseDto(
        job_id="test-job-123",
        status="job_completed",
        page=1,
        total_pages=1,
        page_size=100,
        all_records=[]
    )
    mock_api_instance.jobs.get_job_results.return_value = mock_result
    
    # Test
    client = CatchAllClient(api_key="test_key")
    result = client.search("test query", wait=True)
    
    assert result.job_id == "test-job-123"
    mock_api_instance.jobs.create_job.assert_called_once()
    mock_api_instance.jobs.get_job_status.assert_called()


@patch("langchain_catchall.client.CatchAllApi")
def test_search_without_wait(mock_api_class):
    """Test search method without waiting."""
    # Setup mock
    mock_api_instance = Mock()
    mock_api_class.return_value = mock_api_instance
    
    mock_submit_response = Mock()
    mock_submit_response.job_id = "test-job-123"
    mock_api_instance.jobs.create_job.return_value = mock_submit_response
    
    # Test
    client = CatchAllClient(api_key="test_key")
    result = client.search("test query", wait=False)
    
    assert result.job_id == "test-job-123"
    assert result.status == "pending"
    # Should not call status or results
    mock_api_instance.jobs.get_job_status.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
