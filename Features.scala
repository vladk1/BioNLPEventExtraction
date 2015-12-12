package uk.ac.ucl.cs.mr.statnlpbook.assignment2

import breeze.numerics.abs
import uk.ac.ucl.cs.mr.statnlpbook.assignment2

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

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
    val feats = new mutable.HashMap[FeatureKey, Double]
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
    val feats = new mutable.HashMap[FeatureKey, Double]
    feats += FeatureKey("label bias", List(y)) -> 1.0
    val token = thisSentence.tokens(begin) //first word of argument
    feats += FeatureKey("first argument word", List(token.word, y)) -> 1.0
    feats += FeatureKey("is protein_first trigger word", List(x.isProtein.toString, eventHeadToken.word, y)) -> 1.0
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
    val feats = new mutable.HashMap[FeatureKey, Double]


    feats += FeatureKey("label bias", List(y)) -> 1.0 //bias feature

    // tokens
    addLexicalFeaturesInPlace(feats, thisSentence, begin, end, y)

    // mentions
    addEntityBasedFeaturesInPlace(feats, thisSentence, begin, end, y)

    // dependencies
    addSyntaxBasedFeaturesInPlace(feats, thisSentence, begin, end, x, y)

    feats.toMap
  }

  def addLexicalFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], sentence: Sentence, candBeginInd: Int, candEndInd: Int, y: Label): mutable.HashMap[assignment2.FeatureKey, Double] = {
    val candToken = sentence.tokens(candBeginInd)


    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candBeginInd, y, "candidate token")

    //  add basic token features around candidate
    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candBeginInd - 1, y, "left token from candidate")
    //    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candBeginInd-2, y, "2 left token from candidate")
    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candBeginInd + 1, y, "right token from candidate")
    //    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candBeginInd+1, y, "right token from candidate")

    //  bigrams from candidate
    addNGramPosFeaturesInPlace(feats, sentence.tokens, candBeginInd, y, 2, "right token from candidate")
    addNGramPosFeaturesInPlace(feats, sentence.tokens, candBeginInd - 1, y, 2, "right token from candidate")
    //  threegrams from candidate
    addNGramPosFeaturesInPlace(feats, sentence.tokens, candBeginInd, y, 3, "right token from candidate")
    addNGramPosFeaturesInPlace(feats, sentence.tokens, candBeginInd - 2, y, 3, "right token from candidate")


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

  def addBasicTokenFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], tokens: IndexedSeq[Token], i: Int, y: Label, parent: String) = {
    if (i >= 0 && tokens.size > i) {
      //    lexical string: helps to count general occurrence of particular word as event trigger
      feats += FeatureKey(parent + " bf lexical string", List(tokens(i).word, y)) -> 1.0
      //    stem: helps to count general occurrence of particular modification of word as event trigger (e.g. activate, activates, activating)
      feats += FeatureKey(parent + "bf stem", List(tokens(i).stem, y)) -> 1.0
      //    part-of-speech tag: 98.7 per cent of trigger words are verbs, nouns or adjective
      feats += FeatureKey(parent + "bf pos", List(tokens(i).pos, y)) -> 1.0

      //      didn't help
      //      val hasNumber = tokens(i).word.matches("^[0-9]*$")
      //      val hasUpperCase = tokens(i).word.matches("^[A-Z]*$")
      //      feats += FeatureKey(parent+"bf number", List((hasNumber).toString, y)) -> 1.0
      //      feats += FeatureKey(parent+"bf uppercase", List(hasUpperCase.toString, y)) -> 1.0
    }
  }




  def addNGramPosFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], tokens: IndexedSeq[Token], i: Int, y: Label, nGram: Int, parent: String) = {
    if (i >= 0 && tokens.size > i + nGram) {
      val ngramBasicTokenStrings = getNgramBasicTokenString(i, nGram, tokens)
      feats += FeatureKey(parent + " ngram word", List(ngramBasicTokenStrings._1, y)) -> 1.0
      feats += FeatureKey(parent + " ngram stem", List(ngramBasicTokenStrings._2, y)) -> 1.0
      feats += FeatureKey(parent + " ngram pos", List(ngramBasicTokenStrings._3, y)) -> 1.0
    }
  }

  def getNgramBasicTokenString(i: Int, nGram: Int, tokens: IndexedSeq[Token]) = {
    var a = 0
    var pos, stem, word = ""
    while (a < nGram) {
      pos += (tokens(i + a).pos + " ")
      stem += (tokens(i + a).stem + " ")
      word += (tokens(i + a).word + " ")
      a += 1
    }
    (word, stem, pos)
  }

  def addTokenPOSFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], tokens: IndexedSeq[Token], i: Int, y: Label, featureTemplate: String) = {
    if (i >= 0 && tokens.size > i) {
      feats += FeatureKey(featureTemplate, List(tokens(i).pos, y)) -> 1.0
    }
  }

  def addTokenWordFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], tokens: IndexedSeq[Token], i: Int, y: Label, featureTemplate: String) = {
    if (i >= 0 && tokens.size > i) {
      feats += FeatureKey(featureTemplate, List(tokens(i).word, y)) -> 1.0
    }
  }

  def addAuxilBeforeEventWordFeatInPlace(feats: mutable.HashMap[FeatureKey, Double], tokens: IndexedSeq[Token], i: Int, y: Label, featureTemplate: String) = {
    if (i >= 0 && tokens.size > i) {
      val isAux = tokens(i).pos == "aux"
      val isAuxPass = tokens(i).pos == "auxpass"
      feats += FeatureKey(featureTemplate, List((isAux || isAuxPass).toString, y)) -> 1.0
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
      val dist = Math.abs(mention.begin - candBeginInd)
      (mention.begin > candBeginInd && dist <= LOOK_FORWARD_DIST) ||
        (mention.begin < candBeginInd && dist <= LOOK_BACK_DIST)
    })

    numProteinMentionsAroundCand
  }

  def getNearestDistanceProtein(candSentenceMentions: IndexedSeq[Mention], candBeginInd: Int, toRight: Boolean, toLeft: Boolean) = {
    val proteinDistsAroundCand = candSentenceMentions.map(mention => {
      val dist = Math.abs(mention.begin - candBeginInd)
      if ((mention.begin > candBeginInd && toRight) ||
        (mention.begin < candBeginInd && toLeft)) {
        dist
      }
    }).map(_.toString).filterNot(x => x == "()").map(_.toInt).toIndexedSeq
    if (proteinDistsAroundCand.size > 0) proteinDistsAroundCand.min
    else "-1"
  }


  def addSyntaxBasedFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], sentence: Sentence, candBeginInd: Int, candEndInd: Int,x: Candidate, y: Label) = {
    val candSentenceDeps = sentence.deps
    val candToken = sentence.tokens(candBeginInd)

    // model all syntactic dependency paths up to depth two
    // we extract token features the first and last token in these paths
    val depsMap = new mutable.HashMap[Int, List[(String, Int)]] withDefaultValue Nil
    candSentenceDeps.foreach(dep => {
      depsMap(dep.mod) ::=(dep.label, dep.head)
    })

    //    feats += FeatureKey("dependency map contains candidate", List(depsMap.contains(candBeginInd).toString, y)) -> 1.0

    if (depsMap.contains(candBeginInd)) {
        getAllPaths(1, y, feats, depsMap, sentence, candBeginInd, ArrayBuffer[String](), ArrayBuffer[String]());
    }

  }

  // My argument features
  def getAllPaths(depth: Int,  y: Label, feats: mutable.HashMap[FeatureKey, Double], depsMap: mutable.Map[Int, List[(String, Int)]], sentence: Sentence, idx: Int, posPath: ArrayBuffer[String], edgeLabel: ArrayBuffer[String]): Unit = {

//    println("idx="+idx)
//    println(sentence.tokens)

    if (depsMap.contains(idx) && depth < 5) {
      depsMap(idx).foreach(dep => {
        edgeLabel += dep._1
        val token = sentence.tokens(idx)
        posPath += token.pos
        getAllPaths(depth +1 , y, feats, depsMap, sentence, dep._2, posPath, edgeLabel);
        })
      }

    else{
      //add feature
      feats += FeatureKey("syntactic dependency on edge labels", List(edgeLabel.toString(), posPath.toString(), y)) -> 1.0
//      feats += FeatureKey("syntactic dependency on pos", List(posPath.toString(), y)) -> 1.0
    }
  }

  def myArgumentFeatures(x: Candidate, y: Label): FeatureVector = {
//        println(x)
//        println(y)
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val sentence = doc.sentences(x.sentenceIndex)
    val candSentenceMentions = sentence.mentions
    val event = sentence.events(x.parentIndex) //use this to gain access to the parent event
    val eventHeadToken = sentence.tokens(event.begin) //first token of event
    val token = sentence.tokens(begin) //first word of argument

    val feats = new mutable.HashMap[FeatureKey, Double]

//   bias decreases alottttt!!!!!!!!!!!!!!!!!!!!! we need it-> bcause there are many Nones (f1 however desnt take into account the None label)
    feats += FeatureKey("label bias", List(y)) -> 1.0

    addArgumentLexicalFeatures(x, feats, sentence, begin, end, eventHeadToken, y);
    addArgumentEntityBasedFeaturesInPlace(x, feats,sentence, begin, end, eventHeadToken, y);
    addArgsSyntaxBasedFeatures(feats, sentence, eventHeadToken, x, y)

    feats.toMap
  }

  //  Argument lexical features
  def addArgumentLexicalFeatures(x:Candidate, feats: mutable.HashMap[FeatureKey, Double], sentence: Sentence, candBeginInd: Int, candEndInd: Int, eventTok: Token, y: Label): mutable.HashMap[assignment2.FeatureKey, Double] = {
    val candToken = sentence.tokens(candBeginInd)
    val tokens = sentence.tokens

    if (candBeginInd >= 0 && tokens.size > candBeginInd) {
      //    lexical string: helps to count general occurrence of particular word as arguments
      feats += FeatureKey("lexical feature based on word beg", List(tokens(candBeginInd).word, y)) -> 1.0
      //    stem: helps to count general occurrence of particular modification of word as event trigger (e.g. activate, activates, activating)
      feats += FeatureKey("lexical feature based on stem beg", List(tokens(candBeginInd).stem, y)) -> 1.0
      //    part-of-speech tag: 98.7 per cent of trigger words are verbs, nouns or adjective
      feats += FeatureKey("lexical feature based on pos beg", List(tokens(candBeginInd).pos, y)) -> 1.0
      feats += FeatureKey("lexical feature based on pos event ", List(eventTok.pos,  y)) -> 1.0

      feats += FeatureKey("lexical feature based on length of edge ", List(abs(eventTok.begin - candBeginInd).toString , y)) -> 1.0
      feats += FeatureKey("lexical feature based on - event token", List(eventTok.word.contains("-").toString, x.isProtein.toString, y)) -> 1.0


//      println(getProteinCountArgs(eventTok.begin, candBeginInd, sentence) + "    " + y)
//      feats += FeatureKey("protein mentions between start and end node of the argument edge", List((getProteinCountArgs(eventTok.begin, candBeginInd, sentence)>0).toString(), y)) -> 1.0


      //      didn't help
      //      val hasNumberBeg = tokens(beg).word.matches("^[0-9]*$")
      //      val hasNumberEnd = tokens(end).word.matches("^[0-9]*$")
      //      val hasUpperCase = tokens(end).word.matches("^[A-Z]*$")

      //      feats += FeatureKey("end has number", List(hasNumberBeg.toString, hasNumberEnd.toString, y)) -> 1.0
      //      feats += FeatureKey(parent+"end has uppercase", List(hasUpperCase.toString, y)) -> 1.0
    }

//    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candEndInd, y, "candidate trigger word")
    //  add n-grams (n = 1,2,3) to candidate word
//    addNGramPosFeaturesInPlace(feats, sentence.tokens, candBeginInd, y, 2, "candidate trigger word")
//    addArgsNGramPosFeaturesInPlace(feats, sentence.tokens, eventTok.index, y, 3, "candidate arg trigram word")
    addArgsNGramPosFeaturesInPlace(feats, sentence.tokens, eventTok.index, y, 2, "candidate arg word")

    //  add basic token features around candidate argument word
    //    addBasicTokenFeaturesInPlace(feats, sentence.tokens.first)

    //    //  add basic token features around candidate
    //    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candBeginInd-1, y, "left token from candidate")
    //    //    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candBeginInd-2, y, "2 left token from candidate")
    //    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candBeginInd+1, y, "right token from candidate")
    //    //    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candBeginInd+1, y, "right token from candidate")
    //
    //    //  bigrams from candidate
//        addNGramPosFeaturesInPlace(feats, sentence.tokens, candBeginInd, y, 2, "right token from candidate")
    //    addNGramPosFeaturesInPlace(feats, sentence.tokens, candBeginInd-1, y, 2, "right token from candidate")
    //    //  threegrams from candidate
//        addNGramPosFeaturesInPlace(feats, sentence.tokens, candBeginInd, y, 3, "right token from candidate")
    //    addNGramPosFeaturesInPlace(feats, sentence.tokens, candBeginInd-2, y, 3, "right token from candidate")


    //  Since preposition heads are often indicators of temporal class, we created a new
    //  feature indicating when an event is part of a prepositional phrase. IN=preposition
    //    val isPreposition = candToken.pos != "IN"
    //    feats += FeatureKey("is preposition feature", List(isPreposition.toString, y)) -> 1.0

    //     didnt improve much
    //    Appearance of auxiliaries and modals before the event.
    //    This latter set included all derivations of be and have auxiliaries,
    //    modal words (e.g. may, might, etc.), and the presence/absence of not
    //    addAuxilBeforeEventWordFeatInPlace(feats, sentence.tokens, candBeginInd-1, y, "Auxiliaries presence/absence of not before the event")

    feats
  }

  def addArgumentEntityBasedFeaturesInPlace(x: Candidate, feats: mutable.HashMap[FeatureKey, Double], sentence: Sentence, candBeginInd: Int, candEndInd: Int, eventTok: Token, y: Label) = {

    feats += FeatureKey("Number of prot mentions in candidate's sentence", List(sentence.mentions.size.toString, y)) -> 1.0
//    feats += FeatureKey("is_protein first argument word", List(x.isProtein.toString, eventTok.word, y)) -> 1.0
    feats += FeatureKey("is_protein first argument and stem of event token", List(x.isProtein.toString, eventTok.stem, y)) -> 1.0

//    val nearestProteinDist = getNearestDistanceProtein(candSentenceMentions, candBeginInd, toRight = true, toLeft = true)
//    //    println("nearestProteinDist="+nearestProteinDist)
//    feats += FeatureKey("Nearest protein distance", List(nearestProteinDist.toString, y)) -> 1.0
//    val nearestProteinDistToRight = getNearestDistanceProtein(candSentenceMentions, candBeginInd, toRight = true, toLeft = false)
//    feats += FeatureKey("Nearest protein distance to right", List(nearestProteinDistToRight.toString, y)) -> 1.0
//    val nearestProteinDistToLeft = getNearestDistanceProtein(candSentenceMentions, candBeginInd, toRight = false, toLeft = true)
//    feats += FeatureKey("Nearest protein distance to left", List(nearestProteinDistToLeft.toString, y)) -> 1.0

//    if (sentence.mentions.size > 0) {
//      //       doesnt help
//              val mentionCount = getMentionsAroundCand(sentence.mentions, candBeginInd, 4, 4) // Mentions -a +b around candidate
//              feats += FeatureKey("number of Protein mentions Around candidate", List(mentionCount.size.toString, y)) -> 1.0
//    }

  }


  def addArgsSyntaxBasedFeatures(feats: mutable.HashMap[FeatureKey, Double], sentence: Sentence, eventTok: Token, x: Candidate, y: Label) = {
    val candSentenceDeps = sentence.deps
    val argToken = sentence.tokens(x.begin)
    val eventIdx = eventTok.index;

    // model all syntactic dependency paths up to depth two
    // we extract token features the first and last token in these paths
    val depsMap = new mutable.HashMap[Int, List[(String, Int)]] withDefaultValue Nil
    candSentenceDeps.foreach(dep => {
      depsMap(dep.head) ::= (dep.label, dep.mod)
    })

    if (depsMap.contains(eventIdx)) {
      depsMap.get(eventIdx).get.foreach(d => {
        if(x.begin <= d._2 && d._2 <= x.end){
          feats += FeatureKey("label based dependency", List(d._1, y)) -> 1.0
        }
      })
    }
  }

  def getProteinCountArgs(startIdx:Int, endIdx: Int, sentence: Sentence): Int ={
    val protMentions = sentence.mentions

    if(abs(startIdx - endIdx) > 1){
      var protCount = 0;

      if(startIdx < endIdx){

        for(i <- startIdx+1 to endIdx -1){
          protMentions.foreach(ment => {
            if(ment.begin == i){
              protCount+=1
            }
          })
        }

        protCount

      }
      else{

        for(i <- endIdx+1 to startIdx -1){
          protMentions.foreach(ment => {
            if(ment.begin == i){
              protCount+=1
            }
          })
        }

        protCount

      }
    }
    else 0
  }


  def addArgsNGramPosFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], tokens: IndexedSeq[Token], i: Int, y: Label, nGram: Int, parent: String) = {
    if (i >= 0 && tokens.size > i + nGram) {
      val ngramBasicTokenStrings = getNgramBasicTokenString(i, nGram, tokens)
//      feats += FeatureKey(parent + " ngram word", List(ngramBasicTokenStrings._1, y)) -> 1.0
      feats += FeatureKey(parent + " ngram stem", List(ngramBasicTokenStrings._2, y)) -> 1.0
      feats += FeatureKey(parent + " ngram pos", List(ngramBasicTokenStrings._3, y)) -> 1.0
    }
  }

}
