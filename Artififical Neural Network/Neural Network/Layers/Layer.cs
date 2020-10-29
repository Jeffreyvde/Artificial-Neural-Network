using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Neurons;
using NeuralNetwork.Backpropogation;

namespace NeuralNetwork.Layers
{
    [System.Serializable]
    public abstract class Layer
    {
        public BaseNeuron[] Neurons { get; protected set; }

        /// <summary>
        /// Default constructor for layers
        /// </summary>
        protected Layer() { }

        #region Initialization
        /// <summary>
        /// Generate weights for this layer
        /// </summary>
        /// <param name="previousNeurons">previous neurons to create Connection</param>
        public void ConnectNeurons(BaseNeuron[] nextLayer, BaseNeuron[] previousLayer)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].EstablishConnections(nextLayer, previousLayer);
            }
        }
        #endregion
        #region Training

        /// <summary>
        /// Train this layer
        /// </summary>
        public void FeedForward()
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].FeedForward();
            }
        }

        #endregion
        #region Backpropogation

        /// <summary>
        /// BackPropagate this layer
        /// </summary>
        /// <param name="correctOutputNeuron"></param>
        public void BackPropagate(GradientDescent gradient)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].BackPropagate(gradient);
            }
        }
        #endregion
    }
}
