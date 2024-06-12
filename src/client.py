from flwr.client import Client
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, GetParametersIns, GetParametersRes

from config import DEVICE


class CFLClient(Client):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return super().get_parameters(ins)
    
    def fit(self, ins: FitIns) -> FitRes:
        return super().fit(ins)
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return super().evaluate(ins)
    