#!/usr/bin/env python3
"""
Scenario Manager
Loads and manages scenario configurations from YAML files
"""

import yaml
import os
from typing import Dict, List, Any


class ScenarioManager:
    """Manages scenario configurations"""
    
    def __init__(self, scenarios_dir: str = None):
        """
        Initialize scenario manager
        
        Args:
            scenarios_dir: Path to scenarios directory
        """
        if scenarios_dir is None:
            # Default to config/scenarios/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            scenarios_dir = os.path.join(current_dir, "scenarios")
        
        self.scenarios_dir = scenarios_dir
        self.scenarios = {}
        
    def load_scenario(self, scenario_file: str) -> Dict[str, Any]:
        """
        Load a scenario from YAML file
        
        Args:
            scenario_file: Name of scenario file (e.g., "scenario_I.yaml")
        
        Returns:
            Dictionary containing scenario configuration
        """
        filepath = os.path.join(self.scenarios_dir, scenario_file)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scenario file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            scenario = yaml.safe_load(f)
        
        # Validate scenario structure
        self._validate_scenario(scenario)
        
        return scenario
    
    def load_all_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all scenario files from scenarios directory
        
        Returns:
            Dictionary mapping scenario names to configurations
        """
        if not os.path.exists(self.scenarios_dir):
            raise FileNotFoundError(f"Scenarios directory not found: {self.scenarios_dir}")
        
        scenarios = {}
        for filename in sorted(os.listdir(self.scenarios_dir)):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                scenario = self.load_scenario(filename)
                scenario_name = scenario.get('name', filename.replace('.yaml', '').replace('.yml', ''))
                scenarios[scenario_name] = scenario
        
        self.scenarios = scenarios
        return scenarios
    
    def list_scenarios(self) -> List[str]:
        """
        List all available scenario names
        
        Returns:
            List of scenario names
        """
        if not self.scenarios:
            self.load_all_scenarios()
        
        return list(self.scenarios.keys())
    
    def get_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        Get a specific scenario by name
        
        Args:
            scenario_name: Name of the scenario
        
        Returns:
            Scenario configuration dictionary
        """
        if not self.scenarios:
            self.load_all_scenarios()
        
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found. Available: {self.list_scenarios()}")
        
        return self.scenarios[scenario_name]
    
    def _validate_scenario(self, scenario: Dict[str, Any]):
        """
        Validate scenario structure
        
        Args:
            scenario: Scenario configuration dictionary
        """
        required_fields = ['name', 'clients', 'experiment']
        for field in required_fields:
            if field not in scenario:
                raise ValueError(f"Missing required field '{field}' in scenario")
        
        # Validate clients structure
        clients = scenario['clients']
        for client_type in ['short', 'long', 'mix']:
            if client_type not in clients:
                raise ValueError(f"Missing client type '{client_type}' in scenario")
            
            client_config = clients[client_type]
            if 'count' not in client_config:
                raise ValueError(f"Missing 'count' for client type '{client_type}'")
            if 'qpm' not in client_config:
                raise ValueError(f"Missing 'qpm' for client type '{client_type}'")
            if 'slo' not in client_config:
                raise ValueError(f"Missing 'slo' for client type '{client_type}'")
            
            # Validate list lengths match count
            count = client_config['count']
            if len(client_config['qpm']) != count:
                raise ValueError(f"QPM list length ({len(client_config['qpm'])}) "
                               f"doesn't match count ({count}) for '{client_type}'")
            if len(client_config['slo']) != count:
                raise ValueError(f"SLO list length ({len(client_config['slo'])}) "
                               f"doesn't match count ({count}) for '{client_type}'")
    
    def get_bash_args(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert scenario configuration to bash script arguments
        
        Args:
            scenario: Scenario configuration dictionary
        
        Returns:
            Dictionary of bash script arguments
        """
        clients = scenario['clients']
        experiment = scenario['experiment']
        
        # Convert lists to space-separated strings
        short_qpm = ' '.join(map(str, clients['short']['qpm']))
        long_qpm = ' '.join(map(str, clients['long']['qpm']))
        mix_qpm = ' '.join(map(str, clients['mix']['qpm']))
        
        short_slo = ' '.join(map(str, clients['short']['slo']))
        long_slo = ' '.join(map(str, clients['long']['slo']))
        mix_slo = ' '.join(map(str, clients['mix']['slo']))
        
        return {
            'SHORT_CLIENTS': clients['short']['count'],
            'SHORT_QPM': short_qpm,
            'SHORT_CLIENTS_SLO': short_slo,
            'SHORT_CLIENT_QPM_RATIO': clients['short']['qpm_ratio'],
            
            'LONG_CLIENTS': clients['long']['count'],
            'LONG_QPM': long_qpm,
            'LONG_CLIENTS_SLO': long_slo,
            'LONG_CLIENT_QPM_RATIO': clients['long']['qpm_ratio'],
            
            'MIX_CLIENTS': clients['mix']['count'],
            'MIX_QPM': mix_qpm,
            'MIX_CLIENTS_SLO': mix_slo,
            'MIX_CLIENT_QPM_RATIO': clients['mix']['qpm_ratio'],
            
            'ROUND_NUM': experiment['round_num'],
            'ROUND_TIME': experiment['round_time'],
            'CONCURRENCY': experiment['concurrency'],
            'NUM_REQUESTS': experiment['num_requests'],
            'REQUEST_TIMEOUT': experiment['request_timeout'],
            'SLEEP_TIME': experiment['sleep_time'],
            
            'DISTRIBUTION': scenario.get('distribution', 'normal'),
            'USE_TIME_DATA': scenario.get('use_time_data', 0),
        }


class VLLMConfigManager:
    """Manages vLLM engine configurations"""
    
    def __init__(self, config_file: str = None):
        """
        Initialize vLLM config manager
        
        Args:
            config_file: Path to vLLM config file
        """
        if config_file is None:
            # Default to config/vllm/engine_config.yaml
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(current_dir, "vllm", "engine_config.yaml")
        
        self.config_file = config_file
        self.config = None
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load vLLM configuration from YAML file
        
        Returns:
            Dictionary containing vLLM configuration
        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"vLLM config file not found: {self.config_file}")
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        return self.config
    
    def get_bash_args(self) -> Dict[str, Any]:
        """
        Convert vLLM configuration to bash script arguments
        
        Returns:
            Dictionary of bash script arguments
        """
        if self.config is None:
            self.load_config()
        
        return {
            'MODEL_PATH': self.config['model_path'],
            'TOKENIZER_PATH': self.config['tokenizer_path'],
            'REQUEST_MODEL_NAME': self.config['request_model_name'],
            
            'TENSOR_PARALLEL_SIZE': self.config['tensor_parallel_size'],
            'PIPELINE_PARALLEL_SIZE': self.config['pipeline_parallel_size'],
            
            'GPU_MEMORY_UTILIZATION': self.config['gpu_memory_utilization'],
            'SWAP_SPACE': self.config['swap_space'],
            
            'MAX_MODEL_LEN': self.config['max_model_len'],
            'MAX_NUM_SEQS': self.config['max_num_seqs'],
            'MAX_NUM_BATCHED_TOKENS': self.config['max_num_batched_tokens'],
            
            'DEVICE': self.config['device'],
            'DTYPE': self.config['dtype'],
            'QUANTIZATION': self.config['quantization'],
            
            'TRUST_REMOTE_CODE': str(self.config['trust_remote_code']).lower(),
            'ENABLE_CHUNKED_PREFILL': str(self.config['enable_chunked_prefill']).lower(),
            'DISABLE_LOG_STATS': str(self.config['disable_log_stats']).lower(),
            'ENABLE_PREFIX_CACHING': str(self.config['enable_prefix_caching']).lower(),
            
            'SCHEDULING_POLICY': self.config['scheduling_policy'],
            
            'START_ENGINE': str(self.config['start_engine']).lower(),
            'VLLM_URL': self.config['vllm_url'],
            'API_KEY': self.config['api_key'],
            
            'USE_TUNNEL': self.config['use_tunnel'],
            'LOCAL_PORT': self.config['local_port'],
            'REMOTE_PORT': self.config['remote_port'],
        }


# CLI for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            # List all scenarios
            manager = ScenarioManager()
            scenarios = manager.list_scenarios()
            print("Available scenarios:")
            for scenario in scenarios:
                print(f"  - {scenario}")
        
        elif command == "show" and len(sys.argv) > 2:
            # Show specific scenario
            scenario_name = sys.argv[2]
            manager = ScenarioManager()
            scenario = manager.get_scenario(scenario_name)
            
            print(f"\nScenario: {scenario['name']}")
            print(f"Description: {scenario['description']}")
            print(f"\nClients:")
            for client_type, config in scenario['clients'].items():
                if config['count'] > 0:
                    print(f"  {client_type}: {config['count']} clients")
                    print(f"    QPM: {config['qpm']}")
                    print(f"    SLO: {config['slo']}")
        
        elif command == "vllm":
            # Show vLLM config
            manager = VLLMConfigManager()
            config = manager.load_config()
            print("\nvLLM Engine Configuration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
        
        else:
            print("Usage:")
            print("  python scenario_manager.py list")
            print("  python scenario_manager.py show <scenario_name>")
            print("  python scenario_manager.py vllm")
    
    else:
        print("Usage:")
        print("  python scenario_manager.py list")
        print("  python scenario_manager.py show <scenario_name>")
        print("  python scenario_manager.py vllm")

