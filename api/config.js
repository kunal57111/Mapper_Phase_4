/**
 * Vercel serverless: expose API base URL from env.
 * Set API_URL in Vercel project settings to your backend (e.g. Railway/Render).
 */
module.exports = (req, res) => {
  res.setHeader('Cache-Control', 'public, s-maxage=60, stale-while-revalidate=300');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.status(200).json({
    apiUrl: process.env.API_URL || 'http://127.0.0.1:8000',
  });
};
