from flwr.client import ClientFn

from transformers import AutoModelForCausalLM

from config import DEVICE
from client import CFLClient
from models import Net


def get_client_fn(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    quantization_config=None,
    device_map=None,
    trust_remote_code=None,
    torch_dtype=None,
) -> ClientFn:
    def client_fn(cid: str) -> CFLClient:
        """Create a Flower client representing a single organization."""

        # Load Model
        net = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )

        # Load data
        # Create a  single Flower client representing a single organization
        
    return client_fn


# Virtual Client Engine
def client_fn(cid: str) -> CFLClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)
    net = AutoModelForCausalLM.from_pretrained()

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    # trainloader = trainloaders[int(cid)]
    # valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    # return CFLClient(net, trainloader, valloader).to_client()
