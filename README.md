## NeuroVision

Medical imaging web app for brain tumor segmentation. Built with Next.js, Tailwind CSS, ShadCN-style UI, Framer Motion, and Zustand.

### Features
- Soft dark theme and clean typography
- Landing page with CTA and animations
- Dashboard for MRI upload and mock inference
- Visualization with slice viewer and overlay toggle

### Getting Started
1. Install dependencies:
```bash
npm install
```
2. Run the dev server:
```bash
npm run dev
```
Open `http://localhost:3000`.

### Notes
- File upload currently generates a synthetic 3D volume for demo purposes.
- "Run Segmentation" creates a procedural ellipsoidal mask as a stand-in for a model.

### Tech
- Next.js App Router, TypeScript
- Tailwind CSS
- ShadCN-style primitives (custom lightweight components in `components/ui`)
- Framer Motion for animations
- Zustand for client state
"# brats7d" 
