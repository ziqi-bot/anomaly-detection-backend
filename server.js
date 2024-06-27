



// // Master username
// // peter
// // Master password
// // 19970116Dzq
// // Endpoint
// // database-1.c14e88eiqjuo.us-east-2.rds.amazonaws.com








// require('dotenv').config();

// const express = require('express');
// const bodyParser = require('body-parser');
// const multer = require('multer');
// const cors = require('cors');
// const path = require('path');
// const fs = require('fs');

// // Initialize Express app
// const app = express();
// const PORT = process.env.PORT || 5000;


// // // Initialize PostgreSQL client
// // const pool = new Pool({
// //   user: process.env.PGUSER,
// //   host: process.env.PGHOST,
// //   database: process.env.PGDATABASE,
// //   password: process.env.PGPASSWORD,
// //   port: process.env.PGPORT,
// // });

// // Middleware setup
// app.use(bodyParser.json());
// app.use(bodyParser.urlencoded({ extended: true }));
// app.use(cors());

// const upload = multer();

// // Handle root URL request
// app.get('/', (req, res) => {
//   res.send('Welcome to the results server');
// });

// app.get('/favicon.ico', (req, res) => res.status(204).end());

// // Function to get the latest file from a directory
// const getLatestFilePath = (directory, extension) => {
//   const files = fs.readdirSync(directory).filter(file => file.endsWith(extension));
//   if (files.length === 0) {
//     throw new Error(`No ${extension} file found in ${directory} directory`);
//   }

//   const latestFile = files.map(file => ({
//     name: file,
//     time: fs.statSync(path.join(directory, file)).mtime.getTime()
//   })).sort((a, b) => b.time - a.time)[0].name;

//   return path.join(directory, latestFile);
// };

// // Function to get all files from a directory
// const getAllFilePaths = (directory, extension) => {
//   const files = fs.readdirSync(directory).filter(file => file.endsWith(extension));
//   if (files.length === 0) {
//     throw new Error(`No ${extension} file found in ${directory} directory`);
//   }

//   return files.map(file => path.join(directory, file));
// };

// // Function to check if all detection values are invalid (0, empty, or NaN)
// const areAllDetectionsInvalid = (detections) => {
//   return detections.every(d => d === 0 || d === '' || isNaN(d));
// };

// // Handle saving results
// app.post('/saveResults', upload.none(), (req, res) => {
//   try {
//     console.log('Request Body:', req.body);
//     const detections = JSON.parse(req.body.averageCounts);
//     console.log('Parsed Detections:', detections);

//     if (!Array.isArray(detections) || detections.length === 0) {
//       throw new Error('Detections data is invalid');
//     }

//     // Check if all detection values are invalid
//     if (areAllDetectionsInvalid(detections)) {
//       console.log('All detection values are invalid. Results not saved.');
//       return res.status(200).json({ message: 'All detection values are invalid. Results not saved.' });
//     }

//     const resultsFilePath = path.join(__dirname, 'results', `${Date.now()}-results.txt`);

//     // 将检测结果保存到文件中
//     fs.writeFileSync(resultsFilePath, detections.join('\n'), 'utf8');
//     console.log('Results saved to:', resultsFilePath);

//     res.status(200).json({ message: 'Results saved successfully' });
//   } catch (error) {
//     console.error('Error saving results:', error);
//     res.status(500).json({ error: 'Failed to save results' });
//   }
// });

// // Endpoint to serve the results file content as JSON
// app.get('/api/results', (req, res) => {
//   try {
//     const latestResultsPath = getLatestFilePath('./results', '.txt');
//     const resultsContent = fs.readFileSync(latestResultsPath, 'utf8');
//     res.json({ results: resultsContent.split('\n').filter(line => line.trim() !== '') });
//   } catch (error) {
//     console.error(error.message);
//     res.status(404).json({ error: error.message });
//   }
// });

// // Endpoint to serve all results
// app.get('/api/allResults', (req, res) => {
//   try {
//     const resultsPaths = getAllFilePaths('./results', '.txt');
//     const resultsContent = resultsPaths.map(filePath => {
//       const content = fs.readFileSync(filePath, 'utf8');
//       return { filePath, content: content.split('\n').filter(line => line.trim() !== '') };
//     });
//     res.json({ results: resultsContent });
//   } catch (error) {
//     console.error(error.message);
//     res.status(404).json({ error: error.message });
//   }
// });

// // Start the HTTP server
// const server = app.listen(PORT, () => {
//   console.log(`Server is running on port ${PORT}`);
// });




































































































require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const multer = require('multer');
const cors = require('cors');
const { Pool } = require('pg');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 5000;

// Initialize PostgreSQL client
const pool = new Pool({
  user: process.env.PGUSER,
  host: process.env.PGHOST,
  database: process.env.PGDATABASE,
  password: process.env.PGPASSWORD,
  port: process.env.PGPORT,
});

// Middleware setup
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cors());

const upload = multer();

// Handle root URL request
app.get('/', (req, res) => {
  res.send('Welcome to the results server');
});

app.get('/favicon.ico', (req, res) => res.status(204).end());

// Function to check if all detection values are invalid (0, empty, or NaN)
const areAllDetectionsInvalid = (detections) => {
  return detections.every(d => d === 0 || d === '' || isNaN(d));
};

// Handle saving results
app.post('/saveResults', upload.none(), async (req, res) => {
  try {
    console.log('Request Body:', req.body);
    const detections = JSON.parse(req.body.averageCounts);
    console.log('Parsed Detections:', detections);

    if (!Array.isArray(detections) || detections.length !== 6) {
      throw new Error('Detections data is invalid');
    }

    // Check if all detection values are invalid
    if (areAllDetectionsInvalid(detections)) {
      console.log('All detection values are invalid. Results not saved.');
      return res.status(200).json({ message: 'All detection values are invalid. Results not saved.' });
    }

    // Save detections to database
    const query = 'INSERT INTO results (pedestrian, biker, skater, cart, car, bus) VALUES ($1, $2, $3, $4, $5, $6) RETURNING *';
    const values = detections;
    const result = await pool.query(query, values);

    console.log('Results saved to database:', result.rows[0]);
    res.status(200).json({ message: 'Results saved successfully', data: result.rows[0] });
  } catch (error) {
    console.error('Error saving results:', error);
    res.status(500).json({ error: 'Failed to save results' });
  }
});

// 提供最新结果的端点
app.get('/api/results', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM results ORDER BY created_at DESC LIMIT 1');
    if (result.rows.length === 0) {
      throw new Error('No results found');
    }
    res.json(result.rows[0]);
  } catch (error) {
    console.error(error.message);
    res.status(404).json({ error: error.message });
  }
});

// 提供所有结果的端点
app.get('/api/allResults', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM results ORDER BY created_at DESC');
    res.json(result.rows);
  } catch (error) {
    console.error(error.message);
    res.status(404).json({ error: error.message });
  }
});

// CRUD 操作
// 读取所有结果
app.get('/api/results', async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM results ORDER BY created_at DESC');
    res.json(result.rows);
  } catch (error) {
    console.error(error.message);
    res.status(404).json({ error: error.message });
  }
});

// 读取单个结果
app.get('/api/results/:id', async (req, res) => {
  const { id } = req.params;
  try {
    const result = await pool.query('SELECT * FROM results WHERE id = $1', [id]);
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Result not found' });
    }
    res.json(result.rows[0]);
  } catch (error) {
    console.error(error.message);
    res.status(404).json({ error: error.message });
  }
});

// 更新结果
app.put('/api/results/:id', async (req, res) => {
  const { id } = req.params;
  const { pedestrian, biker, skater, cart, car, bus } = req.body;
  try {
    const result = await pool.query(
      'UPDATE results SET pedestrian = $1, biker = $2, skater = $3, cart = $4, car = $5, bus = $6 WHERE id = $7 RETURNING *',
      [pedestrian, biker, skater, cart, car, bus, id]
    );
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Result not found' });
    }
    res.json(result.rows[0]);
  } catch (error) {
    console.error('Error updating result:', error);
    res.status(500).json({ error: 'Failed to update result' });
  }
});

// 删除结果
app.delete('/api/results/:id', async (req, res) => {
  const { id } = req.params;
  try {
    const result = await pool.query('DELETE FROM results WHERE id = $1 RETURNING *', [id]);
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Result not found' });
    }
    res.json({ message: 'Result deleted successfully', data: result.rows[0] });
  } catch (error) {
    console.error('Error deleting result:', error);
    res.status(500).json({ error: 'Failed to delete result' });
  }
});

// 启动 HTTP 服务器
const server = app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

