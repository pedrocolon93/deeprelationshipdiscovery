import torch
import numpy as np
import torch.optim as optim

config = {
    "layers":3,
    "input_dim":300,
    "hidden_size":2048
}
class Discriminator(torch.nn.Module):
    def __init__(self,config):
        super(Discriminator, self).__init__()
        self.config = config
        self.layers = [torch.nn.Linear(config["hidden_size"],config["hidden_size"]) for _ in range(config["layers"])]
        self.dropout = [torch.nn.Dropout(0.3)for _ in range(config["layers"])]
        self.hidden_activations = [torch.nn.ReLU() for x in range(config["layers"])]
        self.input = torch.nn.Linear(config["input_dim"],config["hidden_size"])
        self.input_activation = torch.nn.ReLU()
        self.output = torch.nn.Linear(config["hidden_size"],1)
        self.output_activation = torch.nn.Sigmoid()

    def forward(self, generation):
        rep = self.input(generation)
        rep = self.input_activation(rep)
        for layer in range(self.config["layers"]):
            rep = self.layers[layer](rep)
            rep = self.hidden_activations[layer](rep)
        rep = self.output(rep)
        rep = self.output_activation(rep)
        return rep

class Generator(torch.nn.Module):
    def __init__(self,config):
        super(Generator, self).__init__()
        self.config = config
        self.dropout = [torch.nn.Dropout(0.3)for _ in range(config["layers"])]
        self.layers = [torch.nn.Linear(config["hidden_size"],config["hidden_size"]) for i in range(config["layers"])]
        self.hidden_activations = [torch.nn.ReLU() for x in range(config["layers"])]
        self.input = torch.nn.Linear(config["input_dim"],config["hidden_size"])
        self.input_activation = torch.nn.ReLU()
        self.output = torch.nn.Linear(config["hidden_size"],config["input_dim"])

    def forward(self, generation):
        rep = self.input(generation)
        rep = self.input_activation(rep)
        for layer in range(self.config["layers"]):
            rep = self.layers[layer](rep)
            rep = self.hidden_activations[layer](rep)
            rep = self.dropout[layer](rep)
        rep = self.output(rep)
        return rep

def train_disciminator(discriminator, inputs, labels, optimizer):
    outputs = discriminator(inputs)
    loss = torch.nn.MSELoss(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


def train_generator(generator, example_input, example_output,optimizer):
    outputs = generator(example_input)
    loss = torch.nn.MSELoss()
    l = loss(outputs, example_output)
    l.backward()
    optimizer.step()
    return l


if __name__ == '__main__':
    discriminator_config = {
        "layers":3,
        "input_dim":300,
        "hidden_size":2048
    }

    discriminator = Discriminator(discriminator_config)
    generator = Generator(discriminator_config)
    example = torch.rand((32,300))
    example_target = torch.randint(0,1,(32,300))
    adam_opt = optim.Adam(generator.parameters())
    epochs = 5
    batches = 100
    for i in range(epochs):
        for j in range(batches):
            # train generator
            example_input = torch.normal(3, 0.5, (32, 300))
            example_output = torch.normal(1, 0.05, (32, 300))
            loss = train_generator(generator, example_input, example_output,adam_opt)
            print(loss)

    exampletest = generator(torch.normal(3, 0.05, (32, 300)))
    print(exampletest)

    # res = discriminator(example)
    #
    # train_disciminator(discriminator)
    # print(res)