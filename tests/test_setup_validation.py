"""Validation tests to verify testing infrastructure is set up correctly."""

import pytest


class TestInfrastructureSetup:
    """Test class to validate the testing infrastructure."""
    
    @pytest.mark.unit
    def test_pytest_installed(self):
        """Verify pytest is available."""
        import pytest
        assert pytest.__version__
    
    @pytest.mark.unit
    def test_coverage_installed(self):
        """Verify pytest-cov is available."""
        import pytest_cov
        assert pytest_cov
    
    @pytest.mark.unit
    def test_mock_installed(self):
        """Verify pytest-mock is available."""
        import pytest_mock
        assert pytest_mock
    
    @pytest.mark.unit
    def test_fixtures_available(self, temp_dir, mock_data, mock_config):
        """Verify custom fixtures are available and working."""
        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test mock_data fixture
        assert mock_data is not None
        assert len(mock_data) == 52  # 52 weeks of data
        assert 'sales' in mock_data.columns
        assert 'media_0' in mock_data.columns
        
        # Test mock_config fixture
        assert isinstance(mock_config, dict)
        assert 'n_media_channels' in mock_config
        assert mock_config['n_media_channels'] == 3
    
    @pytest.mark.unit
    def test_markers_defined(self, request):
        """Verify custom markers are defined."""
        markers = request.config.getini('markers')
        marker_names = [m.split(':')[0].strip() for m in markers]
        assert 'unit' in marker_names
        assert 'integration' in marker_names
        assert 'slow' in marker_names
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        import time
        time.sleep(0.1)  # Simulate slow test
        assert True
    
    def test_project_structure(self):
        """Verify the project structure is accessible."""
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent
        assert project_root.exists()
        assert (project_root / 'lightweight_mmm').exists()
        assert (project_root / 'pyproject.toml').exists()
        assert (project_root / 'tests').exists()
        assert (project_root / 'tests' / 'conftest.py').exists()


def test_basic_assertion():
    """Basic test to ensure pytest runs."""
    assert 1 + 1 == 2


def test_fixture_usage(sample_media_data, sample_target_data):
    """Test that fixtures from conftest.py are accessible."""
    assert sample_media_data.shape == (52, 3)
    assert len(sample_target_data) == 52