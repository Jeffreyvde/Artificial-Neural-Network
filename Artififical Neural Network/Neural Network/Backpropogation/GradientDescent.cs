using System.Collections.Generic;

namespace NeuralNetworks
{
    public class GradientDescent
    {
        private List<GradientDecentValue> steps = new List<GradientDecentValue>();

        /// <summary>
        /// Add a value to gradient decent
        /// </summary>
        public void Add(double step, IBackpropogatable backpropogatable)
        {
            steps.Add(new GradientDecentValue(-step, backpropogatable));
        }

        /// <summary>
        /// Apply gradient decent 
        /// </summary>
        public void Apply()
        {
            for (int i = 0; i < steps.Count; i++)
            {
                steps[i].Apply();
            }
        }
    }
}
