namespace NeuralNetworks
{
    public class TrainingData
    {
        public double[] inputData;
        public int correctOutputNeuron;

        public TrainingData(double[] input, byte correctOuput)
        {
            inputData = input;
            correctOutputNeuron = correctOuput;
        }

        public override string ToString()
        {
            string s = "";
            for (int x = 0; x < 28; ++x)
            {
                for (int y = 0; y < 28; ++y)
                {
                    double value = inputData[x + (y * 28)];
                    if (value == 0)
                        s += " "; // white
                    else if (value == 255)
                        s += "O"; // black
                    else
                        s += "."; // gray
                }
                s += "\n";
            }
            s += correctOutputNeuron;
            return s;
        }
    }
}
