# Setting Up GitHub Pages for Your Slides

Follow these steps to host your slides online using GitHub Pages (it's free!).

## Step 1: Push the Slides to GitHub

```bash
# From your QMUL-Lecture directory
git add slides/
git commit -m "Add interactive lecture slides with Reveal.js"
git push origin main
```

## Step 2: Enable GitHub Pages

1. Go to your repository on GitHub: `https://github.com/[your-username]/QMUL-Lecture`
2. Click on **Settings** (top right, gear icon)
3. In the left sidebar, click **Pages**
4. Under "Build and deployment":
   - **Source:** Deploy from a branch
   - **Branch:** Select `main` (or `master`)
   - **Folder:** Select `/ (root)`
   - Click **Save**

## Step 3: Wait for Deployment

- GitHub will take 1-2 minutes to build and deploy
- You'll see a message: "Your site is live at `https://[username].github.io/QMUL-Lecture/`"

## Step 4: Access Your Slides

Your slides will be available at:
```
https://[your-username].github.io/QMUL-Lecture/slides/
```

## Step 5: Update Your Repository README

Add a link to your slides in your main `README.md`:

```markdown
## 📊 Lecture Slides

Interactive slides for the course lectures:
- [View Slides Online](https://[your-username].github.io/QMUL-Lecture/slides/)
- [Slides Source Code](./slides/)

## 📓 Jupyter Notebooks

- [Bayesian NN MNIST Demo](./bayesian_nn_mnist_demo.ipynb)

## 🚀 Getting Started

[Your existing setup instructions...]
```

## Testing Locally First (Optional)

Before pushing to GitHub, you can test locally:

```bash
# Navigate to slides directory
cd slides

# Start a local server (Python 3)
python -m http.server 8000

# Open in browser:
# http://localhost:8000
```

## Updating Slides

Every time you update the slides:

```bash
# Edit part_a.md or part_b.md
git add slides/
git commit -m "Update lecture content"
git push origin main
```

GitHub Pages will automatically rebuild (takes ~1 minute).

## Sharing on LinkedIn

### Option 1: Share the Live Link
- Post the URL: `https://[username].github.io/QMUL-Lecture/slides/`
- Add a description of your lecture
- Tag relevant topics (#AI #MachineLearning #Education)

### Option 2: Share as PDF
1. Visit your slides with `?print-pdf`:
   `https://[username].github.io/QMUL-Lecture/slides/?print-pdf`
2. Print to PDF (Ctrl/Cmd + P → Save as PDF)
3. Upload the PDF to LinkedIn as a document post

## Troubleshooting

**Issue:** Slides not loading / 404 error
- **Solution:** Wait 2-3 minutes for GitHub Pages to deploy
- Check that GitHub Pages is enabled in Settings
- Verify the URL path includes `/slides/`

**Issue:** Changes not appearing
- **Solution:** Clear browser cache (Ctrl+Shift+R or Cmd+Shift+R)
- Wait 1-2 minutes for GitHub to rebuild
- Check that changes were pushed: `git log`

**Issue:** Math equations not rendering
- **Solution:** Ensure you're using the live URL (not viewing locally without a server)
- Check equation syntax: `$...$` for inline, `$$...$$` for block

## Need Help?

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Reveal.js Documentation](https://revealjs.com/)

---

**Pro Tip:** After setup, you can edit your markdown files directly on GitHub's web interface, and the slides will automatically update!
