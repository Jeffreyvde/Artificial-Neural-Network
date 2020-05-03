using System;
using System.CodeDom;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNetwork.Activations;
using NeuralNetwork.Layers;
using NeuralNetwork.Neurons;
using NeuralNetwork.Tests.Mocks;

namespace NeuralNetwork.Tests
{
    [TestClass]
    public class HiddenNeuronTests
    {
        [TestInitialize]
        public void InitializeHiddenNeuron()
        {

        }


        [TestMethod]
        public void HiddenNeuronDefaultBehaviourTest()
        {
            MockRandom randomMock = new MockRandom();
            Sigmoid sigmoid = new Sigmoid();
            HiddenNeuron neuron = new HiddenNeuron(randomMock, sigmoid);

            Assert.AreEqual(neuron.ActivationFunction, sigmoid);
            Assert.AreEqual(neuron.Bias, randomMock.Range(-1, 1));
            Assert.AreNotEqual(neuron.Bias, randomMock.Range(0, 10));
            Assert.AreEqual(neuron.DerivativeActivation, 0);
            Assert.AreEqual(neuron.DerivativeCost, 0);
            Assert.AreEqual(neuron.Activation, 0);
        }

        [TestMethod]
        public void HiddenNeuronFeedForwardTestNoConnections()
        {
            MockRandom randomMock = new MockRandom();
            Sigmoid sigmoid = new Sigmoid();
            HiddenNeuron neuron = new HiddenNeuron(randomMock, sigmoid);

            Assert.ThrowsException<InvalidOperationException>(neuron.FeedForward);
            Assert.AreEqual(neuron.Activation, 0);
            Assert.AreEqual(neuron.DerivativeActivation, 0);
        }

        [TestMethod]
        public void HiddenNeuronFeedForwardTest()
        {
            MockRandom randomMock = new MockRandom();
            Sigmoid sigmoid = new Sigmoid();
            HiddenNeuron neuron = new HiddenNeuron(randomMock, sigmoid);

            const int size = 9;
            DenseLayer layer = new DenseLayer(size, sigmoid);
            neuron.EstablishConnections(randomMock, null, layer.Neurons);


            neuron.FeedForward();
            Assert.AreEqual(neuron.Activation, sigmoid.CalculateActivation(1));
            Assert.AreEqual(neuron.DerivativeActivation, sigmoid.CalculateDerivativeActivation(1));
        }
    }
}
