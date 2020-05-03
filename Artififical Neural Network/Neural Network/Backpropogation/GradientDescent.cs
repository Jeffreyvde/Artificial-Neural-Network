using System;
using System.Collections.Generic;

namespace NeuralNetwork.Backpropogation
{
    public class GradientDescent
    {
        private readonly List<GradientDecentValue> steps = new List<GradientDecentValue>();

        /// <summary>
        /// Add gradient value to gradient decent
        /// </summary>
        public void Add(double step, IBackpropogatable backpropogatable)
        {
            steps.Add(new GradientDecentValue(step, backpropogatable));
        }

        /// <summary>
        /// Apply gradient decent 
        /// </summary>
        public void Apply(double learningRate)
        {
            for (int i = 0; i < steps.Count; i++)
            {
                steps[i].Apply(learningRate);
            }
        }

        #region ArithmeticOperators


        /// <summary>
        /// + operator for gradient descents
        /// </summary>
        /// <param name="gradient"></param>
        /// <param name="other"></param>
        /// <returns></returns>
        public static GradientDescent operator +(GradientDescent gradient, GradientDescent other)
        {
            if (gradient == null)
                throw new ArgumentNullException(nameof(gradient));
            if (other == null)
                throw new ArgumentNullException(nameof(other));

            GradientDescent newDescent = new GradientDescent();
            for (int i = 0; i < gradient.steps.Count; i++)
            {
                newDescent.Add(gradient.steps[i].StepValue + other.steps[i].StepValue, gradient.steps[i].ChangedObject);
            }

            return newDescent;
        }

        /// <summary>
        /// Gradient descent divide
        /// </summary>
        /// <param name="gradient"></param>
        /// <param name="division">The amount you want to divide it by</param>
        /// <returns></returns>
        public static GradientDescent operator /(GradientDescent gradient, int division)
        {
            if (gradient == null)
                throw new ArgumentNullException(nameof(gradient));

            for (int i = 0; i < gradient.steps.Count; i++)
            {
                GradientDecentValue step = gradient.steps[i];
                step.StepValue /= division;
            }

            return gradient;
        }


        /// <summary>
        /// Gradient descent *
        /// </summary>
        /// <param name="gradient">The gradient value to be multiplied with</param>
        /// <param name="times">The amount you want to multiply it with</param>
        /// <returns></returns>
        public static GradientDescent operator *(GradientDescent gradient, float times)
        {
            if (gradient == null)
                throw new ArgumentNullException(nameof(gradient));

            for (int i = 0; i < gradient.steps.Count; i++)
            {
                gradient.steps[i].StepValue *= times;
            }

            return gradient;
        }
        #endregion

    }
}
