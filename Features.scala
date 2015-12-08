package uk.ac.ucl.cs.mr.statnlpbook.assignment2

import uk.ac.ucl.cs.mr.statnlpbook.assignment2

import scala.collection.mutable

/**
 * Created by Georgios on 05/11/2015.
 */

object Features {

  /**
   * a feature function with two templates w:word,label and l:label.
   * Example for Trigger Exraction
   * @param x
   * @param y
   * @return
   */
  def defaultTriggerFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex) //use this to gain access to the parent sentence
    val feats = new mutable.HashMap[FeatureKey,Double]
    feats += FeatureKey("label bias", List(y)) -> 1.0 //bias feature
    val token = thisSentence.tokens(begin) //first token of Trigger
    feats += FeatureKey("first trigger word", List(token.word, y)) -> 1.0 //word feature
    feats.toMap
  }
  /**
   * a feature function with two templates w:word,label and l:label.
   * Example for Argument Exraction
   * @param x
   * @param y
   * @return
   */
  def defaultArgumentFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex)
    val event = thisSentence.events(x.parentIndex) //use this to gain access to the parent event
    val eventHeadToken = thisSentence.tokens(event.begin) //first token of event
    val feats = new mutable.HashMap[FeatureKey,Double]
    feats += FeatureKey("label bias", List(y)) -> 1.0
    val token = thisSentence.tokens(begin) //first word of argument
    feats += FeatureKey("first argument word", List(token.word, y)) -> 1.0
    feats += FeatureKey("is protein_first trigger word", List(x.isProtein.toString,eventHeadToken.word, y)) -> 1.0
    feats.toMap
  }

  val helperHashMap = new mutable.HashMap[(String, Int), Int] withDefaultValue 0
  //    val sortedDepss = ListMap(helperHashMap.toSeq.sortWith(_._2>_._2):_*)
  //    helperHashMap.foreach(dep => {
  //      println(dep)
  //    })


  //TODO: make your own feature functions
  def myTriggerFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex) //use this to gain access to the parent sentence
    val feats = new mutable.HashMap[FeatureKey,Double]

    feats += FeatureKey("label bias", List(y)) -> 1.0 //bias feature

    // tokens
    addLexicalFeaturesInPlace(feats, thisSentence, begin, end, y)

    // mentions
    addEntityBasedFeaturesInPlace(feats, thisSentence, begin, end, y)

    // dependencies
    addSyntaxBasedFeaturesInPlace(feats, thisSentence, begin, end, y)

    feats.toMap
  }

  def addLexicalFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], sentence: Sentence, candBeginInd: Int, candEndInd: Int, y:Label): mutable.HashMap[assignment2.FeatureKey, Double] = {
    val candToken = sentence.tokens(candBeginInd)

    feats += FeatureKey("first token trigger word", List(candToken.word, y)) -> 1.0
    feats += FeatureKey("first token trigger lemma", List(candToken.stem, y)) -> 1.0

    addTokenPOSFeaturesInPlace(feats, sentence.tokens, candBeginInd-1, y, "POS of prev token")
//    addTokenPOSFeaturesInPlace(feats, sentence.tokens, candBeginInd, y, "POS of cand token") // didn't work result: -3% -2% -4% o_O
    addTokenPOSFeaturesInPlace(feats, sentence.tokens, candEndInd, y, "POS of next token")

//    addNGramPosFeaturesInPlace(feats, sentence.tokens, candBeginInd-2, y, 3, "POS of NGram tokens") //  didnt improve much

//  Since preposition heads are often indicators of temporal class, we created a new
//  feature indicating when an event is part of a prepositional phrase. IN=preposition
    val isPreposition = candToken.pos == "IN"
    feats += FeatureKey("is preposition feature", List(isPreposition.toString, y)) -> 1.0

//     didnt improve much
//    Appearance of auxiliaries and modals before the event.
//    This latter set included all derivations of be and have auxiliaries,
//    modal words (e.g. may, might, etc.), and the presence/absence of not
//    addAuxilBeforeEventWordFeatInPlace(feats, sentence.tokens, candBeginInd-1, y, "Auxiliaries presence/absence of not before the event")

    feats
  }
  def addTokenPOSFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], tokens: IndexedSeq[Token], i:Int, y:Label, featureTemplate:String) = {
    if (i >= 0 && tokens.size > i) {
      feats += FeatureKey(featureTemplate, List(tokens(i).pos, y)) -> 1.0
    }
  }
  def addTokenWordFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], tokens: IndexedSeq[Token], i:Int, y:Label, featureTemplate:String) = {
    if (i >= 0 && tokens.size > i) {
      feats += FeatureKey(featureTemplate, List(tokens(i).word, y)) -> 1.0
    }
  }
  def addAuxilBeforeEventWordFeatInPlace(feats: mutable.HashMap[FeatureKey, Double], tokens: IndexedSeq[Token], i:Int, y:Label, featureTemplate:String) = {
    if (i >= 0 && tokens.size > i) {
        val isAux = tokens(i).pos == "aux"
        val isAuxPass = tokens(i).pos == "auxpass"
        feats += FeatureKey(featureTemplate, List((isAux||isAuxPass).toString, y)) -> 1.0
    }
  }
  def addNGramPosFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], tokens: IndexedSeq[Token], i:Int, y:Label, nGram:Int, featureTemplate:String) = {
    var a = 0
    if (i >= 0 && tokens.size > i+nGram) {
      var tokenPosString = ""
      while(a < nGram) {
        tokenPosString+=(tokens(i+a).pos+" ")
        a+=1
      }
      feats += FeatureKey(featureTemplate, List(tokenPosString, y)) -> 1.0
    }
  }


  def addEntityBasedFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], sentence: Sentence, candBeginInd: Int, candEndInd: Int, y: Label) = {
    val candSentenceMentions = sentence.mentions
    feats += FeatureKey("number of protein mentions in candidate's sentence", List(candSentenceMentions.size.toString, y)) -> 1.0

//    val noProteinMentions = candSentenceMentions.size == 0
//    feats += FeatureKey("no Protein mentions in sentence",List(noProteinMentions.toString, y)) -> 1.0 // didn't help much

//    val candIsProtein = candSentenceMentions.seq.map(_.begin==candBeginInd).size == 1
//    feats += FeatureKey("current candidate is Protein", List(candIsProtein.toString, y)) -> 1.0 // // didn't help much

    // Todo used to not help:
    if (candSentenceMentions.size > 0) {
        val mentionCount = getMentionsAroundCand(candSentenceMentions, candBeginInd, 2, 0) // Mentions -a +b around candidate
        feats += FeatureKey("number of Protein mentions Around candidate", List(mentionCount.size.toString, y)) -> 1.0
    }

  }
  def getMentionsAroundCand(candSentenceMentions: IndexedSeq[Mention], candBeginInd: Int, LOOK_BACK_DIST: Int, LOOK_FORWARD_DIST: Int) = {
    val numProteinMentionsAroundCand = candSentenceMentions.filterNot(mention => {
      val dist = Math.abs(mention.begin-candBeginInd)
      (mention.begin > candBeginInd && dist<=LOOK_FORWARD_DIST) ||
        (mention.begin < candBeginInd && dist<=LOOK_BACK_DIST)
    })

    numProteinMentionsAroundCand
  }


  def addSyntaxBasedFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], sentence: Sentence, candBeginInd: Int, candEndInd: Int, y: Label) = {
    val candSentenceDeps = sentence.deps

    // model all syntactic dependency paths up to depth two
    // we extract token features the first and last token in these paths
    val depsMap = new mutable.HashMap[Int,List[(String, Int)]] withDefaultValue Nil
    candSentenceDeps.foreach(dep => {
      depsMap(dep.head) ::= (dep.label, dep.mod)
    })

    feats += FeatureKey("dependency map contains candidate", List(depsMap.contains(candBeginInd).toString, y)) -> 1.0

    if (depsMap.contains(candBeginInd)) {
      val firstToken = sentence.tokens(candBeginInd)
      val dependencyInfo = syntacticDependencyDepthTokens(2, depsMap, sentence, candBeginInd, List(), "")

      feats += FeatureKey("no syntactic dependency", List((dependencyInfo.size==0).toString, y)) -> 1.0
      if (dependencyInfo.size > 0) {
        dependencyInfo.foreach(depInfo => {
          val lastToken = depInfo._1
          val depPath = depInfo._2
          feats += FeatureKey("syntactic dependency dep path", List(lastToken.stem, depPath, y)) -> 1.0

          feats += FeatureKey("syntactic dependency dep path b", List(firstToken.word, lastToken.stem, depPath, y)) -> 1.0

          feats += FeatureKey("syntactic dependency lemma", List(firstToken.stem, lastToken.stem, y)) -> 1.0

//          feats += FeatureKey("syntactic dependency pos", List(firstToken.pos, lastToken.pos, y)) -> 1.0 // didn't help much
//          feats += FeatureKey("syntactic dependency word", List(firstToken.word, lastToken.word, y)) -> 1.0 // didn't help much

        })
      }
    }

  }
  // Here going through the dependency graph and collecting dependency path + tokens in the end of the path
  def syntacticDependencyDepthTokens(depth: Int, depsMap: mutable.Map[Int, List[(String, Int)]], sentence: Sentence, candBeginInd:Int, finalTokens:List[(Token, String)], depPath:String):List[(Token, String)] = {
      var finalMutableTokens = List[(Token, String)]()
      if (depth==0) {
        (sentence.tokens(candBeginInd),depPath) :: finalTokens
      } else {
        if (depsMap.contains(candBeginInd)) {

          depsMap(candBeginInd).foreach(dep => {
            val newHead = dep._2
            val newDepPath = depPath+" "+dep._1
            finalMutableTokens = finalMutableTokens ++ syntacticDependencyDepthTokens(depth - 1, depsMap, sentence, newHead, List(), newDepPath)
          })
          finalMutableTokens
        }
        finalMutableTokens
      }
  }




  def myArgumentFeatures(x: Candidate, y: Label): FeatureVector = {
    ???
  }


}
