namespace NeuralNetworks
{
    public class TrainingData
    {
        public double[] inputData;
        public int correctOutputNeuron;

        public TrainingData(double[] input, int correctOuput)
        {
            inputData = input;
            correctOutputNeuron = correctOuput;
        }
    }
}
