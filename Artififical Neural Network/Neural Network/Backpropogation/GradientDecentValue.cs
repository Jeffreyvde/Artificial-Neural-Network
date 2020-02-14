namespace NeuralNetwork
{
    public class GradientDecentValue
    {
        public double stepValue;

        public IBackpropogatable changedObject;

        public GradientDecentValue(double stepValue, IBackpropogatable changedObject)
        {
            this.stepValue = stepValue;
            this.changedObject = changedObject;
        }

        public void Apply(double learningRate)
        {
            changedObject.ApplyGradientDecentStep(stepValue, learningRate);
        }
    }
}
