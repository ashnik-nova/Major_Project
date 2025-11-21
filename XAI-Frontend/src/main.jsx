import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import {
  Route,
  RouterProvider,
  createBrowserRouter,
  createRoutesFromElements,
} from "react-router-dom";
import MainLayout from './App.jsx';
import Home from './pages/Home.jsx';
import Classification from './pages/Classification.jsx';

const router = createBrowserRouter(
  createRoutesFromElements(
    <>
      <Route path='/' element={<MainLayout/>}>
        <Route path='' element={<Home/>} />
        <Route path='/classification' element={<Classification/>} />
      </Route>
    </>
  )
)

createRoot(document.getElementById('root')).render(
   <StrictMode>
          <RouterProvider router={router} />
    </StrictMode>
)
