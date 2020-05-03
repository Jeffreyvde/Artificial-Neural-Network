using System;
using NeuralNetwork.Backpropogation;
using NeuralNetwork.Utilities;

namespace NeuralNetwork.Neurons
{
    /// <summary>
    /// This is the class designed for a basic Neuron. It only holds an activation value.
    /// </summary>
    [Serializable]
    public abstract class BaseNeuron
    {
        [NonSerialized] private double activation;

        public double Activation { get => activation; protected set => activation = value; }
        public Connection[] ForwardConnections { get; protected set; }
        public Connection[] BackwardsConnections { get; protected set; }

        /// <summary>
        /// Constructor for the base neuron
        /// </summary>
        /// <param name="activation"></param>
        protected BaseNeuron(double activation)
        {
            Activation = activation;
        }


        /// <summary>
        /// Empty constructor only for child classes (Activation for certain neurons will be set later)
        /// </summary>
        protected BaseNeuron() { }

        /// <summary>
        /// Generate the connections for this neuron
        /// </summary>
        /// <param name="nextLayer">The next layer in the neural network(Value can be null if they don't exist)</param>
        /// <param name="previousLayer">The previous layer in the neural network(Value can be null if they don't exist)</param>
        public void EstablishConnections(BaseNeuron[] nextLayer, BaseNeuron[] previousLayer)
        {
            EstablishConnections(new RandomRange(), nextLayer, previousLayer);
        }

        /// <summary>
        /// Generate the connections for this neuron
        /// </summary>
        /// <param name="random">The random value you want to use</param>
        /// <param name="nextLayer">The next layer in the neural network(Value can be null if they don't exist)</param>
        /// <param name="previousLayer">The previous layer in the neural network(Value can be null if they don't exist)</param>
        public void EstablishConnections(IRandom random, BaseNeuron[] nextLayer, BaseNeuron[] previousLayer)
        {
            if (random is null)
            {
                throw new ArgumentNullException(nameof(random));
            }

            if (nextLayer == null) 
                ForwardConnections = null;
            else
            {
                ForwardConnections = new Connection[nextLayer.Length];
                for (int i = 0; i < nextLayer.Length; i++)
                {
                    ForwardConnections[i] = new Connection(this, nextLayer[i], random);
                }
            }
            if (previousLayer == null) 
                BackwardsConnections = null;
            else
            {
                BackwardsConnections = new Connection[previousLayer.Length];
                for (int i = 0; i < BackwardsConnections.Length; i++)
                {
                    BackwardsConnections[i] = new Connection(this, previousLayer[i], random);
                }
            }
        }

        /// <summary>
        /// Feed forward the Neuron
        /// </summary>
        public abstract void FeedForward();

        /// <summary>
        /// Back propagate the Neuron
        /// </summary>
        public virtual void BackPropagate(GradientDescent descent)
        {
            for (int i = 0; i < ForwardConnections.Length; i++)
            {
                ForwardConnections[i].BackPropagate(descent);
            }
        }
    }
}
