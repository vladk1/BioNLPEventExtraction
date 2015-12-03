package uk.ac.ucl.cs.mr.statnlpbook

/**
 * Created by Georgios on 09/11/2015.
 */

package object assignment2 {

  type Label = String
  type StructuredLabels = (Label,Seq[Label])

  /**
   * A key/subscript index of a feature function. For example, you may have a feature function
   * {{{
   *   f_{word:giants,Baseball}(x,y) = 0 if y != Baseball, otherwise number of times giants appear in x.
   * }}}
   * Here `{word:giants,Baseball}` is a feature key where `word` corresponds to the
   * template of the feature (it could also be `stem` or `label` etc.), and `giants,Baseball` are the
   * template arguments.
   * @param template the template name, `word` in the example above.
   * @param arguments the template arguments, `giants,Baseball` in the above example.
   */
  type FeatureKey = trainers.FeatureKey
  val FeatureKey = trainers.FeatureKey
  //case class FeatureKey(template: String, arguments: Seq[String])

  // global import for precompiled trainers
  val PrecompiledTrainers = trainers.PrecompiledTrainers

  /**
   * A Vector of weights mapping feature keys to real values.
   */
  type Weights = scala.collection.Map[FeatureKey, Double]

  /**
   * A mutable vector of weights mapping feature keys to real values. Useful during training.
   */
  type MutableWeights = scala.collection.mutable.Map[FeatureKey, Double]

  /**
   * A Feature vector, identical to Weights, but introduced to make code more readable.
   */
  type FeatureVector = scala.collection.Map[FeatureKey, Double]

  /**
   * Sparse Dot product between a feature vector and a weight vector.
   * @param f the feature vector.
   * @param w the weight vector.
   * @return `f dot w`
   */
  def dot(f: FeatureVector, w: Weights) = {
    var result = 0.0
    for ((k, v) <- f) result += w(k) * v
    result
  }
  /**
   * In place addition of a feature vector to a weight vector, `w <- w + scale * f`
   * @param f the feature vector to be added.
   * @param w the weight vector to add to.
   * @param scale the scaling factor for the feature vector.
   */
  def addInPlace(f: FeatureVector, w: MutableWeights, scale: Double) = {
    for ((k, v) <- f) w(k) += v * scale
  }



}
