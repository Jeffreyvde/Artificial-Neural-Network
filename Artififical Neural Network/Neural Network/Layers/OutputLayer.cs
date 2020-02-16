using NeuralNetwork.Neurons;

namespace NeuralNetwork.Layers
{
    public class OutputLayer : Layer
    {
        /// <summary>
        /// Initialises the Layer with Hidden neurons
        /// </summary>
        /// <param name="size"></param>
        /// <param name="activatFunction"></param>
        public OutputLayer(int size, IActivation activatFunction)
        {
            Neurons = new OutPutNeuron[size];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new OutPutNeuron(activatFunction);
            }
        }

        /// <summary>
        /// Set the input data for the input neurons
        /// </summary>
        /// <param name="startData"></param>
        public void SetOutput(double[] correctOutput)
        {
            if (correctOutput == null) throw new System.NullReferenceException("correctOutput can not be null: please check Output layer's SetOutput function");
            else if (correctOutput.Length != Neurons.Length) throw new System.ArgumentException("CorrectOutput can not be a different length from the Neurons of this layer: Please check ouput data.");

            for (int i = 0; i < Neurons.Length; i++)
            {
                ((OutPutNeuron)Neurons[i]).SetOutput(correctOutput[i]);
            }
        }
    }
}
