package uk.ac.ucl.cs.mr.statnlpbook.assignment2

import java.io.File

import org.json4s.JsonAST.{JArray, JInt, JString}
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.io.Source
import scala.util.Random

/**
 * Created by Georgios on 02/11/2015.
 */

/**
 * Use BioNLP nmethods getTrainDevDocuments and getTestDocuments to load the data
 */
object BioNLP {
  /**
   * Loads train data and performs train/dev split
   * @param dir directory where train data are, e.g. "./data/assignment2/bionlp/train"
   * @param ratio (1.0-ratio)*100% of the data will be used for development
   * @param num total number of documents to load. Load less data to speed up debugging
   * @return
   */
  def getTrainDevDocuments(dir: String, ratio:Double=0.8,num: Int = Int.MaxValue) = {
    println(s"Loading BioNLP train set from ${dir}")
    val allDocs = JsonReader.readDocuments(dir, num)
    println(s"${allDocs.length} documents loaded")
    val trainLen = (allDocs.length * ratio).toInt
    println(s"${trainLen} documents will be used for training")
    allDocs.splitAt(trainLen)
  }
  /**
   * Loads test data
   * @param dir directory where test data are, e.g. "./data/assignment2/bionlp/test"
   * @return
   */
  def getTestDocuments(dir: String) = {
    println(s"Loading BioNLP test set from ${dir}")
    val allDocs = JsonReader.readDocuments(dir)
    println(s"${allDocs.length} documents loaded")
    allDocs
  }
  def getTrainDocuments(dir: String,num: Int = Int.MaxValue) = {
    println(s"Loading BioNLP train set from ${dir}...")
    val allDocs = JsonReader.readDocuments(dir, num)
    println(s"${allDocs.length} documents loaded")
    allDocs
  }
}

/**
 *
 * @param gold: String, the true label
 * @param begin: Int, token index where candidate begins
 * @param end: Int, token index where candidate ends
 * @param arguments: arguments (if any)
 * @param parentIndex: Int, pointer to parent structure (e.g. event, if argument)
 * @param sentenceIndex: Int, pointer to the sentence containing the candidate
 * @param doc: Document, the document that contains the candidate
 */
case class Candidate(gold: String, begin: Int, end: Int, arguments: Seq[Candidate], parentIndex: Int, sentenceIndex: Int, doc: Document){
  lazy val isProtein = doc.sentences(sentenceIndex).mentions.map(_.begin).contains(begin)
}

case class Document(txt: String, sentences: List[Sentence], fileName:String, rng:Random = new Random(101)) {
  /**
   * Load all trigger candidates for simple classification
   * @param threshold: threshold for random subsampling of "None" labelled triggers (0.02 is a good value, set to 1.0 for no subsampling)
   * @return a list of candidate triggers
   */
  def triggerCandidates(threshold:Double=1.0) = {
    sentences.flatMap(_.events.zipWithIndex.collect{
      case (e,idx) if (subsample(threshold)(e)) => e.copy(doc = this, arguments = e.arguments.map(_.copy(parentIndex=idx)))
    })
  }
  /**
   * Load all argument candidates for simple classification
   * @param threshold threshold for random subsampling of "None" labelled arguments (0.008 is a good value, set to 1.0 for no subsampling)
   * @return list of candidate arguments
   */
  def argumentCandidates(threshold:Double=1.0) = {
    sentences.flatMap(_.events.flatMap(_.arguments.collect{case arg if subsample(threshold)(arg) => arg.copy(doc=this)}))
  }
  /**
   * Load all event (triggers with arguments) candidates for joint classification
   * @param triggerThreshold 0.02 is a good value, set to 1.0 for no subsampling
   * @param argumentThreshold 0.4 is a good value, set to 1.0 for no subsampling
   * @return list of event candidates
   */
  def jointCandidates(triggerThreshold:Double=1.0,argumentThreshold:Double=1.0) = {
    val allJoint = sentences.flatMap(_.events.zipWithIndex.collect{
      case (e,idx) if (subsample(triggerThreshold)(e)) =>
        e.copy(doc = this, arguments = e.arguments.collect {
            case arg if subsample(argumentThreshold)(arg) => arg.copy(doc = this, parentIndex = idx)
        })
      })
    allJoint.filter{e=>e.arguments.length>0}
  }

  private def subsample(threshold: Double)(x:Candidate) = if (x.gold=="None") rng.nextDouble() <= threshold else true
}

/**
  The components of a document, useful for feature engineering
 **/
/**
 * @param mod: Int, index of token that is modifier
 * @param head: Int, index of token that is head of dependency
 * @param label: String, the type of the dependency
 */
case class Dependency(mod: Int, head: Int, label: String)

/**
 * @param label: String, the label of the mention (e.g. Protein)
 * @param begin: Int, the index of the token at the begining of the mention
 * @param end: Int, token index where mention ends
 */
case class Mention(label: String, begin: Int, end: Int)
/**
 * @param index: Int, index of the token in the sentence
 * @param word: the original word
 * @param stem: the word stem
 * @param pos: String, Part-Of-Speech tag
 * @param begin: token index for beginning of token
 * @param end: token index for end of token
 */
case class Token(index: Int, word: String, stem: String, pos: String, begin: Int, end: Int)
/**
 * @param deps: list of dependencies
 * @param events: list of candidate events
 * @param mentions: list of mentions
 * @param tokens: list of tokens in this sentence
 */
case class Sentence(deps: List[Dependency], events: List[Candidate], mentions: IndexedSeq[Mention], tokens: IndexedSeq[Token])


/**
 * From here below, everything is related to low level loading of the data, reading json files, etc.
 */
object Candidate {
  def unlinkedArgument(gold: String, begin: Int, end: Int, parentIndex: Int, sentenceIndex: Int) = {
    Candidate(gold, begin, end, Seq.empty[Candidate], parentIndex, sentenceIndex, Document.empty)
  }
  def unlinkedEvent(gold: String, begin: Int, end: Int, arguments: Seq[Candidate], sentenceIndex: Int) = {
    Candidate(gold, begin, end, arguments, -1, sentenceIndex,  Document.empty)
  }
}
object Document {
  def empty = Document("", List(),"")
}

object JsonReader {
  val rng = new Random(101)

  // json readers
  def readDocuments(dir: String, num: Int = Int.MaxValue) = {
    val alljsons = getAllJsonFiles(dir)
    val jsons = if (num < alljsons.length) alljsons.take(num) else alljsons
    jsons.map(f => parseDocument(f))
  }

  private def getAllJsonFiles(dir: String) = new File(dir).listFiles().filter(_.getName().endsWith(".json")).toSeq

  private def parseDocument(f: File): Document = {
    val json = readJson(f)
    val txt = extractString(json \ "txt")
    val sentences = getList(json \ "sentences").zipWithIndex.map {
      case (s, sentenceIndex) => {
        val deps = getList(s \ "deps").map(dep => extractDependency(dep))
        val allEventCandidates = getList(s \ "eventCandidates").zipWithIndex.map{case (event,idx) => extractEventCandidate(event, sentenceIndex,idx)}
        val eventCandidates = allEventCandidates
          .filter(e => if (e.arguments.length==1) e.arguments.head.begin!=e.begin else true)
          .filter(e => e.arguments.length > 0)
        val mentions = getList(s \ "mentions").map(men => extractMention(men))
        val tokens = getList(s \ "tokens").map(tok => extractToken(tok))
        Sentence(deps, eventCandidates, mentions.toIndexedSeq, tokens.toIndexedSeq)
      }
    }
    Document(txt, sentences, f.getName, rng)
  }

  private def readJson(f: File) = {
    val source = Source.fromFile(f)
    val lines = try source.mkString finally source.close()
    parse(lines)
  }

  private def extractDependency(dep: JValue): Dependency = {
    val mod = extractInt(dep \ "mod")
    val head = extractInt(dep \ "head")
    val label = extractString(dep \ "label")
    Dependency(mod, head, label)
  }

  private def extractEventCandidate(arg: JValue, sentenceIndex: Int, eventIndex:Int): Candidate = {
    val begin = extractInt(arg \ "begin")
    val end = extractInt(arg \ "end")
    val gold = extractString(arg \ "gold")
    val arguments = getList(arg \ "arguments").map{ arg => extractArgument(arg, sentenceIndex, eventIndex) }
    Candidate.unlinkedEvent(gold, begin, end, arguments, sentenceIndex)
  }

  private def extractArgument(arg: JValue, sentenceIndex: Int, eventIndex: Int): Candidate = {
    val begin = extractInt(arg \ "begin")
    val end = extractInt(arg \ "end")
    val gold = extractString(arg \ "gold")
    Candidate.unlinkedArgument(gold, begin, end, eventIndex, sentenceIndex)
  }

  private def extractMention(men: JValue): Mention = {
    val begin = extractInt(men \ "begin")
    val end = extractInt(men \ "end")
    val label = extractString(men \ "label")
    Mention(label, begin, end)
  }

  private def extractToken(tok: JValue): Token = {
    val index = extractInt(tok \ "index")
    val word = extractString(tok \ "word")
    val stem = extractString(tok \ "stem")
    val pos = extractString(tok \ "pos")
    val begin = extractInt(tok \ "begin")
    val end = extractInt(tok \ "end")
    Token(index, word, stem, pos, begin, end)
  }

  private def extractString(v: JValue) = v match {
    case JString(x) => x
    case _ => ""
  }

  private def extractInt(v: JValue) = v match {
    case JInt(x) => x.toInt
    case _ => -1
  }

  private def getList(v: JValue) = v match {
    case JArray(arr) => arr
    case _ => List()
  }
}