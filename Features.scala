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

    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candBeginInd, y, "candidate token")

//  add basic token features around candidate
    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candBeginInd-1, y, "left token from candidate")
    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candBeginInd+1, y, "right token from candidate")

//  bigrams from candidate
    addNGramPosFeaturesInPlace(feats, sentence.tokens, candBeginInd, y, 2, "right token from candidate")
    addNGramPosFeaturesInPlace(feats, sentence.tokens, candBeginInd-1, y, 2, "right token from candidate")
//  threegrams from candidate
    addNGramPosFeaturesInPlace(feats, sentence.tokens, candBeginInd, y, 3, "right token from candidate")
    addNGramPosFeaturesInPlace(feats, sentence.tokens, candBeginInd-2, y, 3, "right token from candidate")


//  Since preposition heads are often indicators of temporal class, we created a new
//  feature indicating when an event is part of a prepositional phrase. IN=preposition
    val isPreposition = candToken.pos != "IN"
    feats += FeatureKey("is preposition feature", List(isPreposition.toString, y)) -> 1.0

//     didnt improve much
//    Appearance of auxiliaries and modals before the event.
//    This latter set included all derivations of be and have auxiliaries,
//    modal words (e.g. may, might, etc.), and the presence/absence of not
//    addAuxilBeforeEventWordFeatInPlace(feats, sentence.tokens, candBeginInd-1, y, "Auxiliaries presence/absence of not before the event")

    feats
  }

  def addBasicTokenFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double],tokens: IndexedSeq[Token], i:Int, y:Label, parent:String) = {
    if (i >= 0 && tokens.size > i) {
      //    lexical string: helps to count general occurrence of particular word as event trigger
      feats += FeatureKey(parent+" bf lexical string", List(tokens(i).word, y)) -> 1.0
      //    stem: helps to count general occurrence of particular modification of word as event trigger (e.g. activate, activates, activating)
      feats += FeatureKey(parent+"bf stem", List(tokens(i).stem, y)) -> 1.0
      //    part-of-speech tag: 98.7 per cent of trigger words are verbs, nouns or adjective
      feats += FeatureKey(parent+"bf pos", List(tokens(i).pos, y)) -> 1.0

//      didn't help
//      val hasNumber = tokens(i).word.matches("^[0-9]*$")
//      val hasUpperCase = tokens(i).word.matches("^[A-Z]*$")
//      feats += FeatureKey(parent+"bf number", List((hasNumber).toString, y)) -> 1.0
//      feats += FeatureKey(parent+"bf uppercase", List(hasUpperCase.toString, y)) -> 1.0
    }
  }
  def addNGramPosFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], tokens: IndexedSeq[Token], i:Int, y:Label, nGram:Int, parent:String) = {
    if (i >= 0 && tokens.size > i + nGram) {
      val ngramBasicTokenStrings = getNgramBasicTokenString(i, nGram, tokens)
      feats += FeatureKey(parent + " ngram word", List(ngramBasicTokenStrings._1, y)) -> 1.0
      feats += FeatureKey(parent + " ngram stem", List(ngramBasicTokenStrings._2, y)) -> 1.0
      feats += FeatureKey(parent + " ngram pos", List(ngramBasicTokenStrings._3, y)) -> 1.0
    }
  }
  def getNgramBasicTokenString(i:Int, nGram:Int, tokens: IndexedSeq[Token]) = {
    var a = 0
    var pos, stem, word =""
    while(a < nGram) {
      pos+=(tokens(i+a).pos+" ")
      stem+=(tokens(i+a).stem+" ")
      word+=(tokens(i+a).word+" ")
      a+=1
    }
    (word, stem, pos)
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



  def addEntityBasedFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], sentence: Sentence, candBeginInd: Int, candEndInd: Int, y: Label) = {
    val candSentenceMentions = sentence.mentions
    feats += FeatureKey("Number of protein mentions in candidate's sentence", List(candSentenceMentions.size.toString, y)) -> 1.0

//    val noProteinMentions = candSentenceMentions.size == 0
//    feats += FeatureKey("no Protein mentions in sentence",List(noProteinMentions.toString, y)) -> 1.0 // didn't help much

//    val candIsProtein = candSentenceMentions.seq.map(_.begin==candBeginInd).size == 1
//    feats += FeatureKey("current candidate is Protein", List(candIsProtein.toString, y)) -> 1.0 // // didn't help much

    val nearestProteinDist = getNearestDistanceProtein(candSentenceMentions, candBeginInd, toRight = true, toLeft = true)
    //    println("nearestProteinDist="+nearestProteinDist)
    feats += FeatureKey("Nearest protein distance", List(nearestProteinDist.toString, y)) -> 1.0
    val nearestProteinDistToRight = getNearestDistanceProtein(candSentenceMentions, candBeginInd, toRight = true, toLeft = false)
    feats += FeatureKey("Nearest protein distance to right", List(nearestProteinDistToRight.toString, y)) -> 1.0
    val nearestProteinDistToLeft = getNearestDistanceProtein(candSentenceMentions, candBeginInd, toRight = false, toLeft = true)
    feats += FeatureKey("Nearest protein distance to left", List(nearestProteinDistToLeft.toString, y)) -> 1.0

    if (candSentenceMentions.size > 0) {
      //       doesnt help
//        val mentionCount = getMentionsAroundCand(candSentenceMentions, candBeginInd, 4, 4) // Mentions -a +b around candidate
//        feats += FeatureKey("number of Protein mentions Around candidate", List(mentionCount.size.toString, y)) -> 1.0
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
  def getNearestDistanceProtein(candSentenceMentions: IndexedSeq[Mention], candBeginInd: Int, toRight:Boolean, toLeft:Boolean) = {
    val proteinDistsAroundCand = candSentenceMentions.map(mention => {
      val dist = Math.abs(mention.begin-candBeginInd)
      if ( (mention.begin > candBeginInd && toRight) ||
        (mention.begin < candBeginInd && toLeft) ) {
        dist
      }
    }).map(_.toString).filterNot(x=> x=="()").map(_.toInt).toIndexedSeq
    if (proteinDistsAroundCand.size > 0) proteinDistsAroundCand.min
    else "-1"
  }


  def addSyntaxBasedFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], sentence: Sentence, candBeginInd: Int, candEndInd: Int, y: Label) = {
    val candSentenceDeps = sentence.deps

    // model all syntactic dependency paths up to depth two
    // we extract token features the first and last token in these paths
    val depsMap = new mutable.HashMap[Int,List[(String, Int)]] withDefaultValue Nil
    candSentenceDeps.foreach(dep => {
      depsMap(dep.head) ::= (dep.label, dep.mod)
    })

//    feats += FeatureKey("dependency map contains candidate", List(depsMap.contains(candBeginInd).toString, y)) -> 1.0

    if (depsMap.contains(candBeginInd)) {
//      defaultDependendencyFeaturesInPlace(feats, sentence, candBeginInd, depsMap, 2, y)
//      defaultDependendencyFeaturesInPlace(feats, sentence, candBeginInd, depsMap, 3, y)
//      defaultDependendencyFeaturesInPlace(feats, sentence, candBeginInd, depsMap, 4, y)
    }

  }
  def defaultDependendencyFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], sentence: Sentence, candBeginInd: Int, depsMap: mutable.Map[Int,List[(String, Int)]], depDepth:Int, y: Label) = {
    val firstToken = sentence.tokens(candBeginInd)
    val twoDepthDependencyInfo = syntacticDependencyDepthTokens(depDepth, depsMap, sentence, candBeginInd, List(), "", "", "")

    feats += FeatureKey("no syntactic dependency", List((twoDepthDependencyInfo.size==0).toString, y)) -> 1.0
    if (twoDepthDependencyInfo.size > 0) {
      twoDepthDependencyInfo.foreach(depInfo => {
        val lastToken = depInfo._1
        val depPath = depInfo._2
        val stemPath = depInfo._3
        val posPath = depInfo._4
//        println(depPath)
//        println(stemPath)
//        println(posPath)

        feats += FeatureKey("syntactic dependency dep path depth="+depDepth, List(depPath, y)) -> 1.0
        feats += FeatureKey("syntactic dependency stem path depth="+depDepth, List(stemPath, y)) -> 1.0
        feats += FeatureKey("syntactic dependency pos path depth="+depDepth, List(posPath, y)) -> 1.0

      })
    }
  }

  // Here going through the dependency graph and collecting dependency path + tokens in the end of the path
  def syntacticDependencyDepthTokens(depth: Int, depsMap: mutable.Map[Int, List[(String, Int)]], sentence: Sentence, candBeginInd:Int, finalTokens:List[(Token, String, String, String)],
                                     depPath:String, stemPath:String, posPath:String):List[(Token, String, String, String)] = {
      val tokens = sentence.tokens
      var finalMutableTokens = List[(Token, String, String, String)]()
      if (depth==0) {
        (sentence.tokens(candBeginInd),depPath, stemPath, posPath) :: finalTokens
      } else {
        if (depsMap.contains(candBeginInd)) {

          depsMap(candBeginInd).foreach(dep => {
            val newHead = dep._2
            val newDepPath = depPath+" "+dep._1
            val curToken = tokens(candBeginInd)
            val newStemPath = stemPath+" "+curToken.stem
            val newPosPath = posPath+" "+curToken.pos
            finalMutableTokens = finalMutableTokens ++ syntacticDependencyDepthTokens(depth - 1, depsMap, sentence, newHead, List(),
              newDepPath, newStemPath, newPosPath)
          })
          finalMutableTokens
        }
        finalMutableTokens
      }
  }




  def myArgumentFeatures(x: Candidate, y: Label): FeatureVector = {
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

    feats += FeatureKey("event Head Token", List(eventHeadToken.word, y)) -> 1.0

    feats.toMap
  }


}
