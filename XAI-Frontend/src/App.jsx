import { useState } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import Header from "./components/Header";
import Footer from "./components/Footer";
import { Outlet } from "react-router-dom";
const MainLayout = () => {
  return (
    <>
      <div className="flex flex-col min-h-screen scrollbar-hidden">
        <Header />
        <main className="grow pt-20 ">
          <Outlet />
        </main>
        <Footer />
      </div>
    </>
  );
};
export default MainLayout;
