package uk.ac.ucl.cs.mr.statnlpbook.assignment2

import breeze.numerics.abs
import uk.ac.ucl.cs.mr.statnlpbook.assignment2

import scala.collection.mutable

/**
 * Created by Georgios on 05/11/2015.
 */

object Features {

  val PROTEIN_TAG = "Protein"

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
//    addSyntaxBasedFeaturesInPlace(feats, thisSentence, begin, end, x, y)
    feats.toMap
  }

  def addLexicalFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], sentence: Sentence, candBeginInd: Int, candEndInd: Int, y: Label): mutable.HashMap[assignment2.FeatureKey, Double] = {
    val candToken = sentence.tokens(candBeginInd)

//    feats += FeatureKey("lexical feature based on - character for trigger", List(candToken.word.contains("-").toString, candToken.stem, y)) -> 1.0

    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candBeginInd, y, "candidate token")

    //  add basic token features around candidate
    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candBeginInd - 1, y, "left token from candidate")
    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candBeginInd + 1, y, "right token from candidate")
    //    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candBeginInd-2, y, "2 left token from candidate")
    //    addBasicTokenFeaturesInPlace(feats, sentence.tokens, candBeginInd+1, y, "right token from candidate")

    //  bigrams from candidate
    addNGramBasicFeaturesInPlace(feats, sentence, candBeginInd, y, 2, "bigram")
    addNGramBasicFeaturesInPlace(feats, sentence, candBeginInd - 1, y, 2, "bigram")
    //  threegrams from candidate
    addNGramBasicFeaturesInPlace(feats, sentence, candBeginInd, y, 3, "3gram")
    addNGramBasicFeaturesInPlace(feats, sentence, candBeginInd - 2, y, 3, "3gram")

    //  Since preposition heads are often indicators of temporal class, we created a new
    //  feature indicating when an event is part of a prepositional phrase. IN=preposition
    val isPreposition = candToken.pos != "IN"
    feats += FeatureKey("is preposition feature", List(isPreposition.toString, y)) -> 1.0

    feats
  }

  val allProteins = List[String]()

  def addBasicTokenFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], tokens: IndexedSeq[Token], i: Int, y: Label, parent: String) = {
    if (i >= 0 && tokens.size > i) {
//      if (sentence.mentions.filter(m=> m.label.contains("Protein")).map(_.begin).contains(i)) {// Protein case
      //    lexical string: helps to count general occurrence of particular word as event trigger
      feats += FeatureKey(parent + " bf lexical string", List(tokens(i).word, y)) -> 1.0
      //    stem: helps to count general occurrence of particular modification of word as event trigger (e.g. activate, activates, activating)
      feats += FeatureKey(parent + "bf stem", List(tokens(i).stem, y)) -> 1.0
      //    part-of-speech tag: 98.7 per cent of trigger words are verbs, nouns or adjective
      feats += FeatureKey(parent + "bf pos", List(tokens(i).pos, y)) -> 1.0
    }
  }


  def addNGramBasicFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], sentence: Sentence, i: Int, y: Label, nGram: Int, parent: String) = {
    val tokens = sentence.tokens
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
//      if (sentence.mentions.filter(m=> m.label.contains("Protein")).map(_.begin).contains(i + a)) {// Protein case
      stem += (tokens(i + a).stem + " ")
      word += (tokens(i + a).word + " ")
      pos += (tokens(i + a).pos + " ")
      a += 1
    }
    (word, stem, pos)
  }


  def addEntityBasedFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], sentence: Sentence, candBeginInd: Int, candEndInd: Int, y: Label) = {
    val candSentenceMentions = sentence.mentions

    feats += FeatureKey("Number of protein mentions in candidate's sentence", List(candSentenceMentions.size.toString, y)) -> 1.0

    val nearestProteinDist = getNearestDistanceProtein(candSentenceMentions, candBeginInd, toRight = true, toLeft = true)
    feats += FeatureKey("Nearest protein distance", List(nearestProteinDist.toString, y)) -> 1.0
    val nearestProteinDistToRight = getNearestDistanceProtein(candSentenceMentions, candBeginInd, toRight = true, toLeft = false)
    feats += FeatureKey("Nearest protein distance to right", List(nearestProteinDistToRight.toString, y)) -> 1.0
    val nearestProteinDistToLeft = getNearestDistanceProtein(candSentenceMentions, candBeginInd, toRight = false, toLeft = true)
    feats += FeatureKey("Nearest protein distance to left", List(nearestProteinDistToLeft.toString, y)) -> 1.0
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
    if (proteinDistsAroundCand.size > 0) {
      proteinDistsAroundCand.min
    }
    else "-1"
  }


  def addSyntaxBasedFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], sentence: Sentence, candBeginInd: Int, candEndInd: Int,x: Candidate, y: Label) = {
    val candSentenceDeps = sentence.deps

    // model all syntactic dependency paths up to depth two
    // we extract token features the first and last token in these paths
    val depsMap = new mutable.HashMap[Int, List[(String, Int)]] withDefaultValue Nil
    val depsReverseMap = new mutable.HashMap[Int, List[(String, Int)]] withDefaultValue Nil
    candSentenceDeps.foreach(dep => {
      depsMap(dep.head) ::=(dep.label, dep.mod)
      depsReverseMap(dep.mod) ::=(dep.label, dep.head)
    })

    feats += FeatureKey("outgoing deps", List(depsMap.contains(candBeginInd).toString, y)) -> 1.0
    if (depsMap.contains(candBeginInd)) {
      jumpThroughAllPaths("depsMap", 2, y, feats, depsMap, sentence, candBeginInd, "", "", "")
    }

    feats += FeatureKey("ingoing deps", List(depsReverseMap.contains(candBeginInd).toString, y)) -> 1.0
    if (depsReverseMap.contains(candBeginInd)) {
      jumpThroughAllPaths("depsReverseMap", 1, y, feats, depsReverseMap, sentence, candBeginInd, "", "", "")
    }
  }

  def jumpThroughAllPaths(parentType:String, depth: Int,  y: Label, feats: mutable.HashMap[FeatureKey, Double], depsMap: mutable.Map[Int, List[(String, Int)]], sentence: Sentence, idx: Int,
                          posPath: String, edgeLabelPath: String, stemPath: String):Unit = {

    if (depsMap.contains(idx) && depth > 0) {
      depsMap(idx).foreach(dep => {
        val newEdgeLabelPath = edgeLabelPath + " " + dep._1

        val token = sentence.tokens(idx)
        val newPosPath = posPath + " " + token.pos

        var newStemPath = "" // NOT USING STEM FOR NOW
        if (sentence.mentions.filter(m => m.label.contains("Protein")).map(_.begin).contains(idx)) {
          newStemPath = stemPath + " " + "[Protein]"
        } else {
          newStemPath = stemPath + " " + token.stem
        }

        // achieve 0.4177
        addBasicTokenFeaturesInPlace(feats, sentence.tokens, idx, y, parentType+"syntactic dependency token depth="+depth)
        feats += FeatureKey(parentType+"syntactic dependency on edge and pos through the path", List(newEdgeLabelPath, newPosPath, y)) -> 1.0
//        feats += FeatureKey(parentType+"syntactic dependency on edge and stem through the path", List(newEdgeLabelPath, newStemPath, y)) -> 1.0

        jumpThroughAllPaths(parentType, depth-1 , y, feats, depsMap, sentence, dep._2, newPosPath, newEdgeLabelPath, newStemPath)
      })
    }
  }

  def myArgumentFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val sentence = doc.sentences(x.sentenceIndex)
    val candSentenceMentions = sentence.mentions
    val event = sentence.events(x.parentIndex) //use this to gain access to the parent event
    val eventHeadToken = sentence.tokens(event.begin) //first token of event
    val token = sentence.tokens(begin) //first word of argument

    val feats = new mutable.HashMap[FeatureKey, Double]
    feats += FeatureKey("label bias", List(y)) -> 1.0

    addArgumentLexicalFeatures(x, feats, sentence, begin, end, eventHeadToken, y);
    addArgumentEntityBasedFeaturesInPlace(x, feats,sentence, begin, end, eventHeadToken, y);
    addArgsSyntaxBasedFeatures(feats, sentence, eventHeadToken, x, y)

    feats.toMap
  }


  def myArgumentFeaturesNB(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val sentence = doc.sentences(x.sentenceIndex)
    val event = sentence.events(x.parentIndex) //use this to gain access to the parent event
    val eventHeadToken = sentence.tokens(event.begin) //first token of event
    val candSentenceMentions = sentence.mentions
    val token = sentence.tokens(begin) //first word of argument
    val tokens = sentence.tokens

    val feats = new mutable.HashMap[FeatureKey, Double]

    feats += FeatureKey("label bias", List(y)) -> 1.0
    feats += FeatureKey("  lexical feature based on stem argument", List(tokens(begin).stem, y)) -> 1.0
    feats += FeatureKey("lexical feature based on pos argument", List(tokens(begin).pos, y)) -> 1.0
    feats += FeatureKey("lexical feature based on pos event trigger candidate ", List(eventHeadToken.pos,  y)) -> 1.0
    feats += FeatureKey("lexical feature based on - event trigger token", List(eventHeadToken.word.contains("-").toString, x.isProtein.toString, y)) -> 1.0

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
      feats += FeatureKey(" lexical feature based on word argument", List(tokens(candBeginInd).word, y)) -> 1.0
        //    stem: helps to count general occurrence of particular modification of word as event trigger (e.g. activate, activates, activating)
      feats += FeatureKey("  lexical feature based on stem argument", List(tokens(candBeginInd).stem, y)) -> 1.0
      //    part-of-speech tag: 98.7 per cent of trigger words are verbs, nouns or adjective
      feats += FeatureKey("lexical feature based on pos argument", List(tokens(candBeginInd).pos, y)) -> 1.0

//      feats += FeatureKey("lexical feature based on pos event trigger candidate ", List(eventTok.pos,  y)) -> 1.0
//      feats += FeatureKey("lexical feature based on length of edge between cand trigger and cand arg", List(abs(eventTok.index - candBeginInd).toString, y)) -> 1.0

      feats += FeatureKey("lexical feature based on - event trigger token", List(eventTok.word.contains("-").toString, x.isProtein.toString, y)) -> 1.0
    }
    addArgsNGramPosFeaturesInPlace(feats,sentence.tokens, eventTok.index, y, 3)
    addArgsNGramPosFeaturesInPlace(feats,sentence.tokens, eventTok.index, y, 2)
    addArgsNGramPosFeaturesInPlace(feats,sentence.tokens, candBeginInd, y, 3)

    feats
  }

  def addArgumentEntityBasedFeaturesInPlace(x: Candidate, feats: mutable.HashMap[FeatureKey, Double], sentence: Sentence, candBeginInd: Int, candEndInd: Int, eventTok: Token, y: Label) = {
//    feats += FeatureKey("Number of prot mentions in candidate's sentence", List(sentence.mentions.size.toString, y)) -> 1.0
    feats += FeatureKey("is_protein first argument and stem of event token", List(x.isProtein.toString, eventTok.stem, y)) -> 1.0
    feats += FeatureKey("protein mentions between argument cand and trigger cand", List(getProteinCountArgs(eventTok.begin, candBeginInd, sentence).toString(), y)) -> 1.0
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
          feats += FeatureKey("feature based on label of dependency", List(d._1, y)) -> 1.0
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


  def addArgsNGramPosFeaturesInPlace(feats: mutable.HashMap[FeatureKey, Double], tokens: IndexedSeq[Token], i: Int, y: Label, nGram: Int) = {
    if (i >= 0 && tokens.size > i + nGram) {
      val ngramBasicTokenStrings = getNgramBasicTokenString(i, nGram, tokens)
      feats += FeatureKey(" ngram stem", List(ngramBasicTokenStrings._2, y)) -> 1.0
      feats += FeatureKey(" ngram pos", List(ngramBasicTokenStrings._3, y)) -> 1.0
    }
  }
}
