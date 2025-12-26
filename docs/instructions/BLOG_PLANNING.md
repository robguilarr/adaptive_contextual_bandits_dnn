**INSTRUCTIONS**:

Act as a Lead AI Research Scientist and Strategic Technical Consultant. Your goal is to produce a high-fidelity methodological breakdown and business case study of a technical project (using provided CONTEXT), specifically tailored for a Master’s Thesis application and a high-level corporate audience (Principal Engineers from Big Tech and Associate Partners from Consulting), reason to highly recommend to use a top-down narrative.

Place your implementation inside `docs/posts` as a markdown (MDX format), the final breakdown should be an extensive and well explained blog post. Take your time to reason about the content to place and do internet searches if needed.

1. Narrative & Tone:

- Tone: Professional, instructional, and peer-reviewed. Eliminate "marketing fluff."
- Energy: Enthusiastic but grounded—highlight the project's successes through the lens of objective data and real-life "Usage Examples."
- Audience: Address senior researchers (expecting rigor/math) and industry leaders (e.g., expecting scalability and ROI).

2. Content Structure:

- Show, Don't Just Tell: Provide a clear "Tech Stack" section and integrate concrete code snippets or LaTeX mathematical formalisms (e.g., optimization objectives or reward functions).
- Business Impact: Explicitly quantify the performance enhancements and cost perspectives. Connect technical improvements (e.g., "reduced latency by 40%") to business outcomes (e.g., "enabling real-time decision-making in high-volume gaming live ops environments").
- Iterative Process: Document the "Attempts and Approach"—explain how specific failures led to architectural pivots, demonstrating critical thinking.

3. Academic Rigor:

- Problem Space: Ground the work by defining the limitations of existing research or traditional industry methods.
- Citations: Every claim or reference to external methods must be cited in APA format: "quoted text" (Author, Year, p. X). Ensure sources include a mix of research papers, official documentation, and industry blogs.

4. Formatting:

- Use a clear hierarchy: Headings, bulleted lists, and tables for data comparison.
- Ensure all mathematical equations are rendered in LaTeX for professional presentation.

**CONTEXT:**

I'm building a python for training contextual bandits, evaluating their performance, and deploying models for personalized recommendation systems. This implementation uses deep neural networks to learn Q-values for contextual multi-armed bandit problems, with a focus on in-app purchase (IAP) optimization. The end-to-end workflow from preprocessing to inference was built with TensorFlow to enable GPU runtime execution across all stages.

I used the follwing papers to base my implementation:
- [Neural Contextual Bandits for Personalized Recommendation](docs/materials/Neural%20Contextual%20Bandits%20for%20Personalized%20Recommendation.pdf)
You can use this papers to quote but dont limt yourself to them only.

I also want to see if any of the next papers can be used to explain potential improvements to this project or research ideas for future implementations (keep in mind to be critical on what can be used and read carefully, from the technical and therical standpoint, your suggested implementation should be tailored to the project context/area and use-case):
- [Scalable Neural Contextual Bandit for Recommender Systems](docs/materials/Scalable%20Neural%20Contextual%20Bandit%20for%20Recommender%20Systems.pdf)
- [ARCLE: THE ABSTRACTION AND REASONING CORPUS LEARNING ENVIRONMENT FOR REINFORCEMENT LEARNING](https://pure.korea.ac.kr/en/publications/arcle-the-abstraction-and-reasoning-corpus-learning-environment-f/)
- [ROIDICE: Offline Return on Investment Maximization for Efficient Decision Making](https://pure.korea.ac.kr/en/publications/roidice-offline-return-on-investment-maximization-for-efficient-d/)
- [Self-supervised Multimodal Graph Convolutional Network for collaborative filtering](https://pure.korea.ac.kr/en/publications/self-supervised-multimodal-graph-convolutional-network-for-collab/)
- [Dual Policy Learning for Aggregation Optimization in Graph Neural Network-based Recommender Systems](http://pure.korea.ac.kr/en/publications/dual-policy-learning-for-aggregation-optimization-in-graph-neural/)
- [AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/)
- [Chip Huyen Blog](https://huyenchip.com/blog/)
- [Learning Deep Learning, NVIDIA’s Magnus Ekman](https://ldlbook.com/)
- [Customer Lifetime Value in Video Games Using Deep Learning and Parametric Models](https://arxiv.org/abs/1811.12799)
Learning Deep Learning, was a huge helper, I used this book to learn about neural nets.

The project summary is located at: `README.md`

The current DNN architecture diagrams are at: `docs/development/ARCHITECTURE.md`

The EXPERIMENTS involved to reach the peak performance are in: `docs/development/EXPERIMENTS_JOURNAL.md`. Ideally I want to back-up with math and research ideas every single decision made.

The project also mentioned a legacy/original code of the of version made by the Google's Firebase team, the technical weakenesses are summarized at: `docs/development/LEGACY_ISSUES.md` and the code is in `legacy_code/training.ipynb`, however don't focus too much on this, the most important is the current approach, not the old one.