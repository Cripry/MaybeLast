import React, { useEffect, useState, useRef } from 'react';
import { Container, Table, TableBody, TableCell, TableHead, TableRow, Paper } from '@mui/material';
import { format } from 'date-fns';

interface ITableData {
  id: number;
  RotorSpeedRpmAvg: number;
  WindSpeedAvg: number;
  ActivePowerAvg: number;
  NacellePositionAvg: number;
  Feature1: number;
  Feature3: number;
  Feature7: number;
  Feature28: number;
  DaySin: number;
  DayCos: number;
  YearSin: number;
  YearCos: number;
  HourSin: number;
  HourCos: number;
  MinuteSin: number;
  MinuteCos: number;

}

const getData = async (take: number = 10): Promise<ITableData[]> => {  // Default value is 10
  const res = await fetch(`http://localhost:3000/api/data?take=${take}`, { // Use template literal to include 'take'
    cache: "no-store",
  });

  if (!res.ok) {
    throw new Error("Failed!");
  }

  return res.json();
}

const About: React.FC = (props) => {
  const [tableData, setTableData] = useState<ITableData[]>([]);
  const [files, setFiles] = useState<File[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await getData();
        setTableData(data);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };

    fetchData();
  }, []);
  
  const handleFilesChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newFiles = event.target.files;
    if (newFiles && newFiles.length > 0) {
      setFiles([newFiles[0]]);
      uploadFile(newFiles[0]);
    } else {
      setFiles([]);
    }
  };

  const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      console.log(data);
      if (data.success) {
        alert('File uploaded successfully.');
        setFiles([]);
      } else {
        alert(data.message);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file.');
    }
  };

  return (
    <div>
      <Container>
        <h1>Wind Turbine Data</h1>
        <Paper elevation={3} style={{ padding: '20px', marginBottom: '20px' }}>
          <h2>Upload Data File</h2>
          <input
            type="file"
            accept=".csv, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, application/vnd.ms-excel"
            onChange={handleFilesChange}
          />
        </Paper>
        <h5>Last 10 Entries</h5>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Rotor Speed</TableCell>
              <TableCell>Wind Speed</TableCell>
              <TableCell>Active Power</TableCell>
              <TableCell>Nacelle Position</TableCell>
              <TableCell>Feature 1</TableCell>
              <TableCell>Feature 3</TableCell>
              <TableCell>Feature 7</TableCell>
              <TableCell>Feature 28</TableCell>
              <TableCell>Day Sin</TableCell>
              <TableCell>Day Cos</TableCell>
              <TableCell>Year Sin</TableCell>
              <TableCell>Year Cos</TableCell>
              <TableCell>Hour Sin</TableCell>
              <TableCell>Hour Cos</TableCell>
              <TableCell>Minute Sin</TableCell>
              <TableCell>Minute Cos</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {tableData.map((item, index) => (
              <TableRow key={index}>
                <TableCell>{item.RotorSpeedRpmAvg.toFixed(2)}</TableCell>
                <TableCell>{item.WindSpeedAvg.toFixed(2)}</TableCell>
                <TableCell>{item.ActivePowerAvg.toFixed(2)}</TableCell>
                <TableCell>{item.NacellePositionAvg.toFixed(2)}</TableCell>
                <TableCell>{item.Feature1.toFixed(2)}</TableCell>
                <TableCell>{item.Feature3.toFixed(2)}</TableCell>
                <TableCell>{item.Feature7.toFixed(2)}</TableCell>
                <TableCell>{item.Feature28.toFixed(2)}</TableCell>
                <TableCell>{item.DaySin.toFixed(2)}</TableCell>
                <TableCell>{item.DayCos.toFixed(2)}</TableCell>
                <TableCell>{item.YearSin.toFixed(2)}</TableCell>
                <TableCell>{item.YearCos.toFixed(2)}</TableCell>
                <TableCell>{item.HourSin.toFixed(2)}</TableCell>
                <TableCell>{item.HourCos.toFixed(2)}</TableCell>
                <TableCell>{item.MinuteSin.toFixed(2)}</TableCell>
                <TableCell>{item.MinuteCos.toFixed(2)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Container>
    </div>
  );
};

export default About;