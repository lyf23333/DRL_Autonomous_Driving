import pytest
import os
import json
from src.trust.trust_interface import TrustInterface

@pytest.fixture
def trust_interface_with_data(tmp_path):
    """Create a trust interface with a temporary data directory"""
    interface = TrustInterface(headless=True)
    interface.data_dir = str(tmp_path / "trust_feedback")
    os.makedirs(interface.data_dir, exist_ok=True)
    return interface

def test_initialization():
    """Test trust interface initialization"""
    interface = TrustInterface(headless=True)
    assert interface.trust_level == 0.5
    assert len(interface.manual_interventions) == 0
    assert len(interface.intervention_timestamps) == 0

def test_trust_level_bounds(trust_interface_with_data):
    """Test trust level boundaries"""
    interface = trust_interface_with_data
    
    # Test upper bound
    for _ in range(20):
        interface.trust_level = min(1.0, interface.trust_level + 0.1)
    assert interface.trust_level <= 1.0
    
    # Test lower bound
    for _ in range(20):
        interface.trust_level = max(0.0, interface.trust_level - 0.1)
    assert interface.trust_level >= 0.0

def test_intervention_recording(trust_interface_with_data):
    """Test manual intervention recording"""
    interface = trust_interface_with_data
    initial_trust = interface.trust_level
    
    interface.record_intervention()
    
    assert len(interface.manual_interventions) == 1
    assert len(interface.intervention_timestamps) == 1
    assert interface.trust_level < initial_trust
    assert interface.manual_interventions[0]['trust_level'] == interface.trust_level

def test_data_saving(trust_interface_with_data):
    """Test trust data saving"""
    interface = trust_interface_with_data
    
    # Record some interventions
    for _ in range(3):
        interface.record_intervention()
    
    # Save data
    interface.save_session_data()
    
    # Check if file exists and contains correct data
    session_file = os.path.join(interface.data_dir, f"trust_data_{interface.session_id}.json")
    assert os.path.exists(session_file)
    
    with open(session_file, 'r') as f:
        data = json.load(f)
        assert data['session_id'] == interface.session_id
        assert len(data['manual_interventions']) == 3
        assert data['intervention_count'] == 3
        assert 0.0 <= data['final_trust_level'] <= 1.0

def test_trust_state_retrieval(trust_interface_with_data):
    """Test getting current trust state"""
    interface = trust_interface_with_data
    
    # Initial state
    state = interface.get_current_trust_state()
    assert state['trust_level'] == interface.trust_level
    assert state['recent_interventions'] == 0
    assert state['time_since_last_intervention'] == float('inf')
    
    # After intervention
    interface.record_intervention()
    state = interface.get_current_trust_state()
    assert state['trust_level'] == interface.trust_level
    assert state['recent_interventions'] == 1
    assert state['time_since_last_intervention'] >= 0

def test_cleanup(trust_interface_with_data, tmp_path):
    """Test cleanup process"""
    interface = trust_interface_with_data
    
    # Record some data
    interface.record_intervention()
    
    # Cleanup
    interface.cleanup()
    
    # Check if data was saved
    session_file = os.path.join(interface.data_dir, f"trust_data_{interface.session_id}.json")
    assert os.path.exists(session_file)

def test_handle_events(trust_interface_with_data):
    """Test event handling"""
    interface = trust_interface_with_data
    initial_trust = interface.trust_level
    
    # Test that the interface stays active
    assert interface.handle_events() is True
    
    # Note: Full event testing would require pygame event simulation,
    # which is complex in a headless environment. The actual event handling
    # should be tested in integration tests. 