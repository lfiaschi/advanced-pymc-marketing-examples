# PyMC Labs Blog Style Guide

## Executive Summary

PyMC Labs blog posts blend technical rigor with business accessibility, targeting data science practitioners who need both theoretical understanding and practical implementation guidance. The writing style prioritizes clarity without sacrificing depth, using progressive disclosure to serve multiple audience levels simultaneously.

## Core Voice & Tone

### Primary Characteristics
- **Professionally conversational**: Authoritative yet approachable, avoiding both academic stiffness and casual informality
- **Solution-oriented**: Lead with problems and pain points before introducing solutions
- **Evidence-based**: Support claims with data, benchmarks, or real-world applications
- **Pragmatically optimistic**: Acknowledge limitations while maintaining confidence in approaches

### Example Tone Comparison

❌ **Too Academic**: "The posterior predictive distribution exhibits heteroskedasticity in the residuals, necessitating a hierarchical variance component."

❌ **Too Casual**: "The model's predictions are kinda all over the place, so we threw in some fancy math to fix it."

✅ **PyMC Labs Style**: "The model's predictions showed varying accuracy across different spend levels—a common challenge in MMM. We addressed this by allowing the variance to adapt based on the data patterns."

## Content Structure

### 1. The Hook Pattern
Every post should open with one of these proven patterns:

**Problem-Solution Arc**
> "Marketing teams spend months building MMMs, only to discover their models can't handle changing market conditions. What if you could transform raw spend data into boardroom strategy in just one day?"

**Direct Value Proposition**
> "Unlock the power of marketing analytics with PyMC-Marketing—the open source solution for smarter decision-making."

**Relatable Pain Point**
> "You've collected months of marketing data. You've tried three different MMM libraries. Yet you're still explaining to stakeholders why the model says to spend $0 on your best-performing channel."

### 2. Progressive Disclosure Structure

```
1. Executive Summary (1-2 paragraphs)
   - What problem does this solve?
   - Who should care?
   - What's the key insight?

2. Context & Background (2-3 paragraphs)
   - Industry challenge or theoretical foundation
   - Why existing solutions fall short
   - Bridge to proposed approach

3. Core Content (60-70% of post)
   - Technical implementation
   - Mathematical concepts (when necessary)
   - Code examples or visual demonstrations
   - Results and validation

4. Practical Implications (2-3 paragraphs)
   - Business impact
   - Implementation considerations
   - Scaling potential

5. Call to Action
   - Next steps for interested readers
   - Links to resources
   - Engagement opportunities
```

### 3. Section Transitions

Use rhetorical questions or declarative bridges:
- "But how does this work in practice?"
- "This raises an important question about scalability."
- "Let's dive into the technical implementation."
- "With the theory established, we can explore the results."

## Technical Content Guidelines

### Mathematical Notation

**When to Include Math:**
- Core algorithm explanations requiring precision
- Novel contributions or modifications
- Comparisons between approaches

**How to Present Math:**
1. Introduce concept in plain language first
2. Present the equation with LaTeX formatting
3. Define each parameter immediately after
4. Provide intuitive interpretation

**Example:**
> "We model customer acquisition cost (CAC) as changing over time, rather than assuming it stays constant. Mathematically, we express this as:
>
> `y[t] = S · tanh(x[t]/(S · cac_0[t]))`
>
> Here, `S` represents the maximum saturation level, while `cac_0[t]` captures how the channel's effectiveness changes at time `t`. Think of it as the 'resistance' the channel faces in acquiring new customers."

### Code Integration

**Preferred Approaches:**

1. **Visual Code Screenshots** for API demonstrations
   - Shows simplicity without overwhelming detail
   - Include only essential lines
   - Add annotations if necessary

2. **Linked Notebooks** for complete implementations
   - "See our complete implementation in this [example notebook](link)"
   - Keeps blog post focused on concepts

3. **Inline Code** only for critical snippets
   ```python
   # Keep snippets under 10 lines
   model = MMM(
       data=df,
       channels=['tv', 'radio', 'online'],
       adstock='geometric'
   )
   ```

### Data Visualization

**Requirements:**
- Every chart must have a clear caption explaining the insight
- Use consistent color schemes aligned with PyMC Labs branding
- Annotate key findings directly on visualizations
- Prefer interactive elements when possible (for web versions)

**Caption Template:**
> "Figure 3: Comparison of actual vs. predicted customer acquisitions across three channels. Note how the model captures the seasonal pattern (shaded regions) while maintaining uncertainty bounds."

## Handling Complexity

### The Scaffolding Approach

Build understanding incrementally:

1. **Intuitive Explanation**: "Gaussian Processes help us model patterns that change over time"
2. **Practical Analogy**: "Think of it like a flexible ruler that can bend to match your data's shape"
3. **Technical Detail**: "GPs define a distribution over functions, allowing us to model uncertainty in functional relationships"
4. **Implementation Note**: "In PyMC, we implement this using `gp.Latent` with a Matérn kernel"

### Acknowledging Limitations

Be transparent about constraints:
- "While this approach handles most scenarios, it assumes..."
- "Our benchmarks show strong performance, though results may vary with..."
- "We're glossing over some implementation details here—see our technical documentation for..."

## Stylistic Best Practices

### Language Patterns

**Use Active Voice**
- ✅ "The model identifies three key drivers"
- ❌ "Three key drivers were identified by the model"

**Prefer Specific to General**
- ✅ "Processing 10,000 observations takes 3.2 seconds"
- ❌ "The algorithm is fast"

**Balance Technical and Business Language**
- ✅ "The 95% credible intervals give stakeholders confidence bounds for ROI estimates"
- ❌ "The posterior HPD shows heterogeneous effects"

### Metaphors and Analogies

PyMC Labs posts effectively use metaphors to make abstract concepts concrete:

**Effective Metaphors:**
- "MMM copilot" (AI as assistant, not replacement)
- "validation black holes" (time-consuming debugging)
- "assembling a plane mid-flight" (complex integration challenges)

**Guidelines:**
- Use metaphors that resonate with business/technical hybrid audience
- Avoid mixing metaphors within a section
- Ensure metaphors clarify rather than decorate

### Word Choice and Phrasing

**Preferred Terms:**
- "Leverage" → "Use" or "Apply"
- "Utilize" → "Use"
- "Stakeholders" for business audience
- "Practitioners" for technical audience
- "Insights" for results with business impact
- "Findings" for technical observations

**Avoid:**
- Excessive superlatives ("revolutionary," "game-changing")
- Defensive language ("obviously," "clearly")
- Academic hedging ("it might be suggested that")

## Competitive Positioning

When comparing to alternatives:

1. **Lead with empirical evidence**, not assertions
2. **Acknowledge competitor strengths** where legitimate
3. **Use data visualizations** to support claims
4. **Provide reproducible benchmarks** when possible
5. **Frame as "different tools for different needs"** rather than winner-takes-all

**Example:**
> "Our benchmarks show PyMC-Marketing processes this dataset 3.5x faster than Alternative X, though Alternative X offers more built-in visualization options that some teams may prioritize."

## Calls to Action

### Hierarchy of CTAs

1. **Primary (end of post)**: Direct engagement
   - "Schedule a consultation to discuss your MMM needs"
   - "Try PyMC-Marketing with our quickstart guide"

2. **Secondary (mid-post)**: Resource exploration
   - "Explore our example notebooks"
   - "See the full benchmark results"

3. **Tertiary (contextual)**: Community building
   - "Join the discussion on our forum"
   - "Follow our updates on social media"

### CTA Style
- Consultative rather than salesy
- Focus on value exchange, not transaction
- Include specific next steps

## Common Pitfalls to Avoid

### Technical Writing Traps

❌ **Over-explaining basics**: Don't define Bayesian inference in every post
❌ **Under-explaining innovations**: Do explain what makes your approach novel
❌ **Code dumps**: Large code blocks without context
❌ **Math without intuition**: Equations that don't connect to practical understanding
❌ **Assuming universal knowledge**: Undefined acronyms or jargon

### Tone Missteps

❌ **Overselling**: "The only solution you'll ever need"
❌ **False modesty**: "Our little experiment might be useful"
❌ **Academic pretension**: Unnecessary complexity to sound smart
❌ **Casual dismissiveness**: Treating competitor limitations as fatal flaws

### Structural Issues

❌ **Burying the lede**: Waiting too long to state value proposition
❌ **Kitchen sink approach**: Including every possible detail
❌ **Weak conclusions**: Ending without clear next steps
❌ **Orphaned sections**: Content that doesn't flow with narrative

## Quality Checklist

Before publishing, ensure your post:

### Content
- [ ] Opens with a compelling hook
- [ ] Clearly identifies the target audience
- [ ] Balances technical depth with accessibility
- [ ] Includes concrete examples or case studies
- [ ] Acknowledges limitations appropriately
- [ ] Provides actionable takeaways

### Structure
- [ ] Follows progressive disclosure pattern
- [ ] Uses clear section headings
- [ ] Maintains logical flow between sections
- [ ] Includes appropriate visualizations
- [ ] Ends with clear CTA

### Style
- [ ] Maintains consistent voice throughout
- [ ] Uses active voice predominantly
- [ ] Defines technical terms on first use
- [ ] Avoids excessive jargon
- [ ] Includes relevant links to resources

### Technical Accuracy
- [ ] Code examples are tested and functional
- [ ] Mathematical notation is correct
- [ ] Claims are supported by evidence
- [ ] Benchmarks are reproducible
- [ ] References are properly cited

## Example Blog Post Outline

**Title**: "Reducing Customer Acquisition Costs by 30%: A Bayesian Approach to Marketing Channel Optimization"

**Hook**: "Your marketing team just increased spend by 50% but conversions only grew by 10%. Sound familiar? Here's how Company X used Bayesian MMM to identify and fix channel saturation."

**Structure**:
1. The CAC Challenge (problem identification)
2. Why Traditional MMMs Fall Short (context)
3. Time-Varying Effectiveness: The Missing Piece (solution introduction)
4. Implementation with PyMC-Marketing (technical detail)
5. Results: 30% CAC Reduction in 90 Days (validation)
6. Scaling This Approach (practical application)
7. Get Started Today (CTA)

## Final Recommendations

The most effective PyMC Labs blog posts:

1. **Tell a story** that resonates with practitioner pain points
2. **Build credibility** through transparent methodology
3. **Balance promotion with education** (60/40 ratio)
4. **Use progressive disclosure** to serve multiple audiences
5. **Provide clear next steps** for engaged readers
6. **Maintain technical rigor** without sacrificing accessibility

Remember: The goal is to position PyMC Labs as the thoughtful, evidence-based choice for teams serious about Bayesian modeling, while remaining approachable to those just starting their journey.