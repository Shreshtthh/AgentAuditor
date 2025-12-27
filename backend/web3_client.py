"""
Web3 client for interacting with Cortensor contracts on Arbitrum Sepolia
"""
import json
from pathlib import Path
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
from typing import Dict, Any, List, Tuple
import logging

from backend.config import settings

logger = logging.getLogger(__name__)


class CortensorWeb3Client:
    """Client for interacting with Cortensor SessionV2 and SessionQueueV2 contracts"""
    
    def __init__(self):
        # Connect to Arbitrum Sepolia
        self.w3 = Web3(Web3.HTTPProvider(settings.arbitrum_sepolia_rpc_url))
        
        # Add PoA middleware for Arbitrum compatibility
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        # Load account from private key
        self.account = Account.from_key(settings.private_key)
        self.w3.eth.default_account = self.account.address
        
        # Load contract ABIs
        abi_dir = Path(__file__).parent.parent
        with open(abi_dir / "SessionV2ABI.json", "r") as f:
            session_v2_abi = json.load(f)
        with open(abi_dir / "SessionQueueV2ABI.json", "r") as f:
            session_queue_v2_abi = json.load(f)
        
        # Initialize contract instances
        self.session_v2 = self.w3.eth.contract(
            address=Web3.to_checksum_address(settings.session_v2_address),
            abi=session_v2_abi
        )
        self.session_queue_v2 = self.w3.eth.contract(
            address=Web3.to_checksum_address(settings.session_queue_v2_address),
            abi=session_queue_v2_abi
        )
        
        logger.info(f"Web3 client initialized. Connected: {self.w3.is_connected()}")
        logger.info(f"Account address: {self.account.address}")
    
    def create_session(
        self,
        session_name: str,
        model: str,
        redundant: int = 3,
        num_validators: int = 0,
        task_timeout: int = 120
    ) -> Tuple[int, str]:
        """
        Create a new Cortensor session
        
        Args:
            session_name: Human-readable session name
            model: Model ID (e.g., 'cts-llm-2')
            redundant: Number of redundant nodes for PoI (default: 3)
            num_validators: Number of validators for PoUW (default: 0)
            task_timeout: Task timeout in seconds
        
        Returns:
            Tuple of (session_id, transaction_hash)
        """
        try:
            # Prepare transaction
            tx = self.session_v2.functions.create(
                session_name,
                model,
                self.account.address,
                redundant,
                num_validators,
                task_timeout
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 500000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Parse SessionCreated event to get session ID
            session_created_event = self.session_v2.events.SessionCreated().process_receipt(receipt)
            session_id = session_created_event[0]['args']['sessionId']
            
            logger.info(f"Session created: ID={session_id}, TX={tx_hash.hex()}")
            return session_id, tx_hash.hex()
        
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    def submit_task(self, session_id: int, prompt: str, model_params: Dict[str, Any] = None) -> Tuple[int, str]:
        """
        Submit a task to an existing session
        
        Args:
            session_id: Session ID to submit task to
            prompt: The prompt/input for the task
            model_params: Optional model parameters (temperature, max_tokens, etc.)
        
        Returns:
            Tuple of (task_id, transaction_hash)
        """
        try:
            # Default model parameters
            if model_params is None:
                model_params = {
                    'temperature': 0.7,
                    'max_tokens': 2048,
                    'top_p': 0.9
                }
            
            # Build transaction
            tx = self.session_v2.functions.submit(
                session_id,
                prompt
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 300000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Sign and send
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Parse TaskSubmitted event
            task_submitted_event = self.session_v2.events.TaskSubmitted().process_receipt(receipt)
            task_id = task_submitted_event[0]['args']['taskId']
            
            logger.info(f"Task submitted: Session={session_id}, Task={task_id}, TX={tx_hash.hex()}")
            return task_id, tx_hash.hex()
        
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise
    
    def get_task_results(self, session_id: int, task_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve all results for a task (from all redundant nodes)
        
        Args:
            session_id: Session ID
            task_id: Task ID
        
        Returns:
            List of result dictionaries with node info and outputs
        """
        try:
            # Call getTaskResults function
            results = self.session_queue_v2.functions.getTaskResults(session_id, task_id).call()
            
            # Parse results into structured format
            parsed_results = []
            for i, result in enumerate(results):
                parsed_results.append({
                    'node_index': i,
                    'output': result,  # The actual text output from the node
                    'session_id': session_id,
                    'task_id': task_id
                })
            
            logger.info(f"Retrieved {len(parsed_results)} results for Session={session_id}, Task={task_id}")
            return parsed_results
        
        except Exception as e:
            logger.error(f"Failed to get task results: {e}")
            raise
    
    def get_session_info(self, session_id: int) -> Dict[str, Any]:
        """Get detailed session information"""
        try:
            session = self.session_v2.functions.getSession(session_id).call()
            return {
                'session_id': session_id,
                'name': session[0],
                'model': session[1],
                'owner': session[2],
                'redundant': session[3],
                'num_validators': session[4],
                'active': session[5]
            }
        except Exception as e:
            logger.error(f"Failed to get session info: {e}")
            raise
    
    def deactivate_session(self, session_id: int) -> str:
        """Deactivate a session after use"""
        try:
            tx = self.session_v2.functions.deactivateSession(session_id).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 100000,
                'gasPrice': self.w3.eth.gas_price
            })
            
            signed_tx = self.w3.eth.account.sign_transaction(tx, self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(f"Session {session_id} deactivated: TX={tx_hash.hex()}")
            return tx_hash.hex()
        
        except Exception as e:
            logger.error(f"Failed to deactivate session: {e}")
            raise


# Global client instance
web3_client = CortensorWeb3Client()
