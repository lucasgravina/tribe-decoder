// Background service worker for TRIBE Decoder

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "tribe-analyze-selection",
    title: "🧠 TRIBE Decode: What reaction is this targeting?",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "tribe-analyze-selection" && info.selectionText) {
    // Send selected text to popup via storage, then open popup
    chrome.storage.local.set({ 
      pendingText: info.selectionText,
      source: "selection"
    }, () => {
      chrome.action.openPopup();
    });
  }
});

// Handle messages from content script or popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "ANALYZE_TEXT") {
    analyzeWithTRIBE(request.text, request.apiKey)
      .then(result => sendResponse({ success: true, data: result }))
      .catch(err => sendResponse({ success: false, error: err.message }));
    return true; // keep channel open for async
  }
});

async function analyzeWithTRIBE(text, apiKey) {
  const systemPrompt = `You are a neural reaction decoder inspired by Meta's TRIBE v2 brain encoding model. 

TRIBE v2 maps stimuli to fMRI brain responses across functional regions. Your job is the INVERSE: given text content, decode what emotional and neurological reactions it is engineered to trigger.

Analyze using these neuroscience-grounded brain regions from TRIBE v2's taxonomy:
- **TPJ (Temporoparietal Junction)**: Emotional processing, social cognition, us-vs-them framing
- **FFA (Fusiform Face Area)**: Identity/face-based persuasion, personalization, tribal recognition  
- **Default Mode Network**: Self-referential thinking, personal identity triggers, nostalgia
- **Amygdala**: Fear response, threat detection, urgency, outrage
- **Nucleus Accumbens**: Reward anticipation, dopamine hooks, validation seeking
- **Broca's Area**: Authority through language complexity/simplicity, rhetoric patterns
- **Anterior Insula**: Disgust, moral violations, visceral discomfort
- **V5 (Motion Area)**: Urgency, FOMO, momentum framing

Return a JSON object with this exact structure:
{
  "primary_target": "The #1 emotion/reaction being targeted (1-3 words, punchy)",
  "manipulation_score": 0-10 (how engineered/intentional the emotional targeting feels),
  "brain_regions": [
    {
      "region": "Region name",
      "activation": "low|medium|high",
      "technique": "specific technique being used",
      "example": "brief quote or element from the text that activates this"
    }
  ],
  "dominant_techniques": ["technique1", "technique2", "technique3"],
  "intended_outcome": "What behavior/belief/feeling the content wants to produce",
  "who_benefits": "Who benefits if you react as intended",
  "inoculation": "One sentence on how to mentally defend against this",
  "verdict": "2-3 sentence plain-language summary of what's happening"
}

Only include brain regions that are actually activated (skip ones with no signal). Be precise, not alarmist. Neutral/benign content should get low manipulation scores.`;

  const response = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01"
    },
    body: JSON.stringify({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1500,
      system: systemPrompt,
      messages: [
        {
          role: "user",
          content: `Decode the neural targeting in this content:\n\n${text.slice(0, 3000)}`
        }
      ]
    })
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.error?.message || `API error ${response.status}`);
  }

  const data = await response.json();
  const raw = data.content[0].text;
  
  // Strip markdown fences if present
  const cleaned = raw.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
  return JSON.parse(cleaned);
}
