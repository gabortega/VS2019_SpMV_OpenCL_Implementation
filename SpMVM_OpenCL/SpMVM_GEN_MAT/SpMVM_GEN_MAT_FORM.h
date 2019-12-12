#pragma once

#include<compiler_config.h>

#include<util_misc.hpp>
#include<IO/generateMatrix.hpp>
#include<IO/mmio.h>
#include<IO/convert_input.h>
#include<msclr\marshal_cppstd.h>

namespace SpMVMGENMAT {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for SpMVM_GEN_MAT_FORM
	/// </summary>
	public ref class SpMVM_GEN_MAT_FORM : public System::Windows::Forms::Form
	{
	public:
		SpMVM_GEN_MAT_FORM(void)
		{
			InitializeComponent();
			//
			this->textBox1->Text = (gcnew System::String(INPUT_FOLDER + "/" + GENERATOR_FOLDER + "/"));
			this->textBox2->Text = (gcnew System::String(std::string("new_matrix" + getTimeOfRun()).c_str()));
			//
			//TODO: Add the constructor code here
			//
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~SpMVM_GEN_MAT_FORM()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::GroupBox^ groupBox1;
	protected:



	private: System::Windows::Forms::GroupBox^ groupBox3;
	private: System::Windows::Forms::Label^ label3;

	private: System::Windows::Forms::GroupBox^ groupBox4;


	private: System::Windows::Forms::Label^ label2;
	private: System::Windows::Forms::Label^ label1;
	private: System::Windows::Forms::Label^ label5;


	private: System::Windows::Forms::Label^ label4;
	private: System::Windows::Forms::TextBox^ textBox2;
	private: System::Windows::Forms::TextBox^ textBox1;
	private: System::Windows::Forms::GroupBox^ groupBox5;


	private: System::Windows::Forms::Label^ label6;
	private: System::Windows::Forms::Label^ label7;
	private: System::Windows::Forms::GroupBox^ groupBox2;
	private: System::Windows::Forms::GroupBox^ groupBox6;


	private: System::Windows::Forms::Label^ label8;
	private: System::Windows::Forms::Label^ label9;
	private: System::Windows::Forms::GroupBox^ groupBox7;


	private: System::Windows::Forms::Label^ label10;
	private: System::Windows::Forms::Label^ label11;
	private: System::Windows::Forms::GroupBox^ groupBox8;
	private: System::Windows::Forms::Button^ button4;
	private: System::Windows::Forms::Button^ button3;
	private: System::Windows::Forms::Button^ button2;
	private: System::Windows::Forms::GroupBox^ groupBox9;
	private: System::Windows::Forms::Button^ button6;
	private: System::Windows::Forms::Button^ button7;
	private: System::Windows::Forms::FolderBrowserDialog^ folderBrowserDialog1;
	private: System::Windows::Forms::NumericUpDown^ numericUpDown5;
	private: System::Windows::Forms::NumericUpDown^ numericUpDown4;
	private: System::Windows::Forms::NumericUpDown^ numericUpDown2;
	private: System::Windows::Forms::NumericUpDown^ numericUpDown1;
	private: System::Windows::Forms::NumericUpDown^ numericUpDown3;
	private: System::Windows::Forms::NumericUpDown^ numericUpDown9;
	private: System::Windows::Forms::NumericUpDown^ numericUpDown8;
	private: System::Windows::Forms::NumericUpDown^ numericUpDown7;
	private: System::Windows::Forms::NumericUpDown^ numericUpDown6;

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->groupBox8 = (gcnew System::Windows::Forms::GroupBox());
			this->button4 = (gcnew System::Windows::Forms::Button());
			this->button3 = (gcnew System::Windows::Forms::Button());
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->groupBox5 = (gcnew System::Windows::Forms::GroupBox());
			this->numericUpDown5 = (gcnew System::Windows::Forms::NumericUpDown());
			this->numericUpDown4 = (gcnew System::Windows::Forms::NumericUpDown());
			this->label6 = (gcnew System::Windows::Forms::Label());
			this->label7 = (gcnew System::Windows::Forms::Label());
			this->groupBox4 = (gcnew System::Windows::Forms::GroupBox());
			this->numericUpDown2 = (gcnew System::Windows::Forms::NumericUpDown());
			this->numericUpDown1 = (gcnew System::Windows::Forms::NumericUpDown());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->groupBox3 = (gcnew System::Windows::Forms::GroupBox());
			this->numericUpDown3 = (gcnew System::Windows::Forms::NumericUpDown());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->textBox2 = (gcnew System::Windows::Forms::TextBox());
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
			this->groupBox9 = (gcnew System::Windows::Forms::GroupBox());
			this->button6 = (gcnew System::Windows::Forms::Button());
			this->button7 = (gcnew System::Windows::Forms::Button());
			this->groupBox6 = (gcnew System::Windows::Forms::GroupBox());
			this->numericUpDown9 = (gcnew System::Windows::Forms::NumericUpDown());
			this->numericUpDown8 = (gcnew System::Windows::Forms::NumericUpDown());
			this->label8 = (gcnew System::Windows::Forms::Label());
			this->label9 = (gcnew System::Windows::Forms::Label());
			this->groupBox7 = (gcnew System::Windows::Forms::GroupBox());
			this->numericUpDown7 = (gcnew System::Windows::Forms::NumericUpDown());
			this->numericUpDown6 = (gcnew System::Windows::Forms::NumericUpDown());
			this->label10 = (gcnew System::Windows::Forms::Label());
			this->label11 = (gcnew System::Windows::Forms::Label());
			this->folderBrowserDialog1 = (gcnew System::Windows::Forms::FolderBrowserDialog());
			this->groupBox1->SuspendLayout();
			this->groupBox8->SuspendLayout();
			this->groupBox5->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown5))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown4))->BeginInit();
			this->groupBox4->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown2))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown1))->BeginInit();
			this->groupBox3->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown3))->BeginInit();
			this->groupBox2->SuspendLayout();
			this->groupBox9->SuspendLayout();
			this->groupBox6->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown9))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown8))->BeginInit();
			this->groupBox7->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown7))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown6))->BeginInit();
			this->SuspendLayout();
			// 
			// groupBox1
			// 
			this->groupBox1->Controls->Add(this->groupBox8);
			this->groupBox1->Controls->Add(this->groupBox5);
			this->groupBox1->Controls->Add(this->groupBox4);
			this->groupBox1->Location = System::Drawing::Point(13, 125);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(399, 209);
			this->groupBox1->TabIndex = 1;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"Gaussian Method";
			// 
			// groupBox8
			// 
			this->groupBox8->Controls->Add(this->button4);
			this->groupBox8->Controls->Add(this->button3);
			this->groupBox8->Controls->Add(this->button2);
			this->groupBox8->Location = System::Drawing::Point(6, 143);
			this->groupBox8->Name = L"groupBox8";
			this->groupBox8->Size = System::Drawing::Size(387, 56);
			this->groupBox8->TabIndex = 2;
			this->groupBox8->TabStop = false;
			this->groupBox8->Text = L"Create matrix";
			// 
			// button4
			// 
			this->button4->Location = System::Drawing::Point(249, 20);
			this->button4->Name = L"button4";
			this->button4->Size = System::Drawing::Size(85, 23);
			this->button4->TabIndex = 2;
			this->button4->Text = L"Full";
			this->button4->UseVisualStyleBackColor = true;
			this->button4->Click += gcnew System::EventHandler(this, &SpMVM_GEN_MAT_FORM::button4_Click);
			// 
			// button3
			// 
			this->button3->Location = System::Drawing::Point(144, 20);
			this->button3->Name = L"button3";
			this->button3->Size = System::Drawing::Size(85, 23);
			this->button3->TabIndex = 1;
			this->button3->Text = L"Column-only";
			this->button3->UseVisualStyleBackColor = true;
			this->button3->Click += gcnew System::EventHandler(this, &SpMVM_GEN_MAT_FORM::button3_Click);
			// 
			// button2
			// 
			this->button2->Location = System::Drawing::Point(37, 20);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(85, 23);
			this->button2->TabIndex = 0;
			this->button2->Text = L"Row-only";
			this->button2->UseVisualStyleBackColor = true;
			this->button2->Click += gcnew System::EventHandler(this, &SpMVM_GEN_MAT_FORM::button2_Click);
			// 
			// groupBox5
			// 
			this->groupBox5->Controls->Add(this->numericUpDown5);
			this->groupBox5->Controls->Add(this->numericUpDown4);
			this->groupBox5->Controls->Add(this->label6);
			this->groupBox5->Controls->Add(this->label7);
			this->groupBox5->Location = System::Drawing::Point(6, 81);
			this->groupBox5->Name = L"groupBox5";
			this->groupBox5->Size = System::Drawing::Size(387, 56);
			this->groupBox5->TabIndex = 1;
			this->groupBox5->TabStop = false;
			this->groupBox5->Text = L"Columns";
			// 
			// numericUpDown5
			// 
			this->numericUpDown5->Location = System::Drawing::Point(294, 19);
			this->numericUpDown5->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100000, 0, 0, 0 });
			this->numericUpDown5->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->numericUpDown5->Name = L"numericUpDown5";
			this->numericUpDown5->Size = System::Drawing::Size(76, 20);
			this->numericUpDown5->TabIndex = 3;
			this->numericUpDown5->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
			// 
			// numericUpDown4
			// 
			this->numericUpDown4->Location = System::Drawing::Point(53, 19);
			this->numericUpDown4->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100000, 0, 0, 0 });
			this->numericUpDown4->Name = L"numericUpDown4";
			this->numericUpDown4->Size = System::Drawing::Size(76, 20);
			this->numericUpDown4->TabIndex = 1;
			this->numericUpDown4->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
			// 
			// label6
			// 
			this->label6->AutoSize = true;
			this->label6->Location = System::Drawing::Point(189, 22);
			this->label6->Name = L"label6";
			this->label6->Size = System::Drawing::Size(99, 13);
			this->label6->TabIndex = 2;
			this->label6->Text = L"Standard deviation:";
			// 
			// label7
			// 
			this->label7->AutoSize = true;
			this->label7->Location = System::Drawing::Point(7, 22);
			this->label7->Name = L"label7";
			this->label7->Size = System::Drawing::Size(40, 13);
			this->label7->TabIndex = 0;
			this->label7->Text = L"Mean: ";
			// 
			// groupBox4
			// 
			this->groupBox4->Controls->Add(this->numericUpDown2);
			this->groupBox4->Controls->Add(this->numericUpDown1);
			this->groupBox4->Controls->Add(this->label2);
			this->groupBox4->Controls->Add(this->label1);
			this->groupBox4->Location = System::Drawing::Point(6, 19);
			this->groupBox4->Name = L"groupBox4";
			this->groupBox4->Size = System::Drawing::Size(387, 56);
			this->groupBox4->TabIndex = 0;
			this->groupBox4->TabStop = false;
			this->groupBox4->Text = L"Rows";
			// 
			// numericUpDown2
			// 
			this->numericUpDown2->Location = System::Drawing::Point(294, 20);
			this->numericUpDown2->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100000, 0, 0, 0 });
			this->numericUpDown2->Minimum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 1, 0, 0, 0 });
			this->numericUpDown2->Name = L"numericUpDown2";
			this->numericUpDown2->Size = System::Drawing::Size(76, 20);
			this->numericUpDown2->TabIndex = 3;
			this->numericUpDown2->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
			// 
			// numericUpDown1
			// 
			this->numericUpDown1->Location = System::Drawing::Point(53, 19);
			this->numericUpDown1->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100000, 0, 0, 0 });
			this->numericUpDown1->Name = L"numericUpDown1";
			this->numericUpDown1->Size = System::Drawing::Size(76, 20);
			this->numericUpDown1->TabIndex = 1;
			this->numericUpDown1->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 10, 0, 0, 0 });
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(189, 22);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(99, 13);
			this->label2->TabIndex = 2;
			this->label2->Text = L"Standard deviation:";
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(7, 22);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(40, 13);
			this->label1->TabIndex = 0;
			this->label1->Text = L"Mean: ";
			// 
			// groupBox3
			// 
			this->groupBox3->Controls->Add(this->numericUpDown3);
			this->groupBox3->Controls->Add(this->label5);
			this->groupBox3->Controls->Add(this->label4);
			this->groupBox3->Controls->Add(this->textBox2);
			this->groupBox3->Controls->Add(this->textBox1);
			this->groupBox3->Controls->Add(this->label3);
			this->groupBox3->Location = System::Drawing::Point(13, 13);
			this->groupBox3->Name = L"groupBox3";
			this->groupBox3->Size = System::Drawing::Size(399, 106);
			this->groupBox3->TabIndex = 0;
			this->groupBox3->TabStop = false;
			this->groupBox3->Text = L"General";
			// 
			// numericUpDown3
			// 
			this->numericUpDown3->Location = System::Drawing::Point(141, 18);
			this->numericUpDown3->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100000, 0, 0, 0 });
			this->numericUpDown3->Name = L"numericUpDown3";
			this->numericUpDown3->Size = System::Drawing::Size(76, 20);
			this->numericUpDown3->TabIndex = 1;
			this->numericUpDown3->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 500, 0, 0, 0 });
			// 
			// label5
			// 
			this->label5->AutoSize = true;
			this->label5->Location = System::Drawing::Point(6, 20);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(129, 13);
			this->label5->TabIndex = 0;
			this->label5->Text = L"Matrix dimensions (N x N):";
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Location = System::Drawing::Point(6, 77);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(92, 13);
			this->label4->TabIndex = 4;
			this->label4->Text = L"Output File Name:";
			// 
			// textBox2
			// 
			this->textBox2->Location = System::Drawing::Point(104, 74);
			this->textBox2->Name = L"textBox2";
			this->textBox2->Size = System::Drawing::Size(289, 20);
			this->textBox2->TabIndex = 5;
			// 
			// textBox1
			// 
			this->textBox1->Location = System::Drawing::Point(104, 48);
			this->textBox1->Name = L"textBox1";
			this->textBox1->ReadOnly = true;
			this->textBox1->Size = System::Drawing::Size(289, 20);
			this->textBox1->TabIndex = 3;
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(6, 51);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(74, 13);
			this->label3->TabIndex = 2;
			this->label3->Text = L"Output Folder:";
			// 
			// groupBox2
			// 
			this->groupBox2->Controls->Add(this->groupBox9);
			this->groupBox2->Controls->Add(this->groupBox6);
			this->groupBox2->Controls->Add(this->groupBox7);
			this->groupBox2->Location = System::Drawing::Point(13, 340);
			this->groupBox2->Name = L"groupBox2";
			this->groupBox2->Size = System::Drawing::Size(399, 209);
			this->groupBox2->TabIndex = 2;
			this->groupBox2->TabStop = false;
			this->groupBox2->Text = L"Skip Method";
			// 
			// groupBox9
			// 
			this->groupBox9->Controls->Add(this->button6);
			this->groupBox9->Controls->Add(this->button7);
			this->groupBox9->Location = System::Drawing::Point(6, 142);
			this->groupBox9->Name = L"groupBox9";
			this->groupBox9->Size = System::Drawing::Size(387, 56);
			this->groupBox9->TabIndex = 2;
			this->groupBox9->TabStop = false;
			this->groupBox9->Text = L"Create matrix";
			// 
			// button6
			// 
			this->button6->Location = System::Drawing::Point(203, 19);
			this->button6->Name = L"button6";
			this->button6->Size = System::Drawing::Size(85, 23);
			this->button6->TabIndex = 1;
			this->button6->Text = L"Column-only";
			this->button6->UseVisualStyleBackColor = true;
			this->button6->Click += gcnew System::EventHandler(this, &SpMVM_GEN_MAT_FORM::button6_Click);
			// 
			// button7
			// 
			this->button7->Location = System::Drawing::Point(98, 19);
			this->button7->Name = L"button7";
			this->button7->Size = System::Drawing::Size(85, 23);
			this->button7->TabIndex = 0;
			this->button7->Text = L"Row-only";
			this->button7->UseVisualStyleBackColor = true;
			this->button7->Click += gcnew System::EventHandler(this, &SpMVM_GEN_MAT_FORM::button7_Click);
			// 
			// groupBox6
			// 
			this->groupBox6->Controls->Add(this->numericUpDown9);
			this->groupBox6->Controls->Add(this->numericUpDown8);
			this->groupBox6->Controls->Add(this->label8);
			this->groupBox6->Controls->Add(this->label9);
			this->groupBox6->Location = System::Drawing::Point(6, 81);
			this->groupBox6->Name = L"groupBox6";
			this->groupBox6->Size = System::Drawing::Size(387, 56);
			this->groupBox6->TabIndex = 1;
			this->groupBox6->TabStop = false;
			this->groupBox6->Text = L"Columns";
			// 
			// numericUpDown9
			// 
			this->numericUpDown9->Location = System::Drawing::Point(249, 20);
			this->numericUpDown9->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100000, 0, 0, 0 });
			this->numericUpDown9->Name = L"numericUpDown9";
			this->numericUpDown9->Size = System::Drawing::Size(76, 20);
			this->numericUpDown9->TabIndex = 3;
			this->numericUpDown9->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 0 });
			// 
			// numericUpDown8
			// 
			this->numericUpDown8->Location = System::Drawing::Point(46, 20);
			this->numericUpDown8->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100000, 0, 0, 0 });
			this->numericUpDown8->Name = L"numericUpDown8";
			this->numericUpDown8->Size = System::Drawing::Size(76, 20);
			this->numericUpDown8->TabIndex = 1;
			// 
			// label8
			// 
			this->label8->AutoSize = true;
			this->label8->Location = System::Drawing::Point(189, 22);
			this->label8->Name = L"label8";
			this->label8->Size = System::Drawing::Size(54, 13);
			this->label8->TabIndex = 2;
			this->label8->Text = L"Skip step:";
			// 
			// label9
			// 
			this->label9->AutoSize = true;
			this->label9->Location = System::Drawing::Point(8, 22);
			this->label9->Name = L"label9";
			this->label9->Size = System::Drawing::Size(32, 13);
			this->label9->TabIndex = 0;
			this->label9->Text = L"Start:";
			// 
			// groupBox7
			// 
			this->groupBox7->Controls->Add(this->numericUpDown7);
			this->groupBox7->Controls->Add(this->numericUpDown6);
			this->groupBox7->Controls->Add(this->label10);
			this->groupBox7->Controls->Add(this->label11);
			this->groupBox7->Location = System::Drawing::Point(6, 19);
			this->groupBox7->Name = L"groupBox7";
			this->groupBox7->Size = System::Drawing::Size(387, 56);
			this->groupBox7->TabIndex = 0;
			this->groupBox7->TabStop = false;
			this->groupBox7->Text = L"Rows";
			// 
			// numericUpDown7
			// 
			this->numericUpDown7->Location = System::Drawing::Point(249, 20);
			this->numericUpDown7->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100000, 0, 0, 0 });
			this->numericUpDown7->Name = L"numericUpDown7";
			this->numericUpDown7->Size = System::Drawing::Size(76, 20);
			this->numericUpDown7->TabIndex = 3;
			this->numericUpDown7->Value = System::Decimal(gcnew cli::array< System::Int32 >(4) { 5, 0, 0, 0 });
			// 
			// numericUpDown6
			// 
			this->numericUpDown6->Location = System::Drawing::Point(46, 20);
			this->numericUpDown6->Maximum = System::Decimal(gcnew cli::array< System::Int32 >(4) { 100000, 0, 0, 0 });
			this->numericUpDown6->Name = L"numericUpDown6";
			this->numericUpDown6->Size = System::Drawing::Size(76, 20);
			this->numericUpDown6->TabIndex = 1;
			// 
			// label10
			// 
			this->label10->AutoSize = true;
			this->label10->Location = System::Drawing::Point(189, 22);
			this->label10->Name = L"label10";
			this->label10->Size = System::Drawing::Size(54, 13);
			this->label10->TabIndex = 2;
			this->label10->Text = L"Skip step:";
			// 
			// label11
			// 
			this->label11->AutoSize = true;
			this->label11->Location = System::Drawing::Point(8, 22);
			this->label11->Name = L"label11";
			this->label11->Size = System::Drawing::Size(32, 13);
			this->label11->TabIndex = 0;
			this->label11->Text = L"Start:";
			// 
			// SpMVM_GEN_MAT_FORM
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(424, 561);
			this->Controls->Add(this->groupBox2);
			this->Controls->Add(this->groupBox3);
			this->Controls->Add(this->groupBox1);
			this->Name = L"SpMVM_GEN_MAT_FORM";
			this->Text = L"Generate Matrix";
			this->groupBox1->ResumeLayout(false);
			this->groupBox8->ResumeLayout(false);
			this->groupBox5->ResumeLayout(false);
			this->groupBox5->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown5))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown4))->EndInit();
			this->groupBox4->ResumeLayout(false);
			this->groupBox4->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown2))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown1))->EndInit();
			this->groupBox3->ResumeLayout(false);
			this->groupBox3->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown3))->EndInit();
			this->groupBox2->ResumeLayout(false);
			this->groupBox9->ResumeLayout(false);
			this->groupBox6->ResumeLayout(false);
			this->groupBox6->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown9))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown8))->EndInit();
			this->groupBox7->ResumeLayout(false);
			this->groupBox7->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown7))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->numericUpDown6))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion
	private: System::Void generateMatrixImage(struct coo_t* coo)
	{
		System::Drawing::Bitmap image(coo->n, coo->n);
		msclr::interop::marshal_context context;
		std::string gen_filename = (INPUT_FOLDER + (std::string)"/" + GENERATOR_FOLDER + (std::string)"/" + context.marshal_as<std::string>(this->textBox2->Text) + ".bmp");
		//
		for (IndexType i = 0; i < coo->nnz; i++)
		{
			image.SetPixel(coo->jc[i] - 1, coo->ir[i] - 1, System::Drawing::Color::FromArgb(255, 1, 1));
		}
		image.Save(gcnew System::String(gen_filename.c_str()), System::Drawing::Imaging::ImageFormat::Bmp);
	}

	private: System::Void button2_Click(System::Object^ sender, System::EventArgs^ e)
	{
		FILE* f;
		struct coo_t coo;
		//
		msclr::interop::marshal_context context;
		std::string gen_filename = (INPUT_FOLDER + (std::string)"/" + GENERATOR_FOLDER + (std::string)"/" + context.marshal_as<std::string>(this->textBox2->Text) + ".mtx");
		if (createOutputDirectory(INPUT_FOLDER, GENERATOR_FOLDER))
		{
			showOutputDirErrorMessage();
			return;
		}
		//
		generateMatrixGaussMethodRow(System::Decimal::ToUInt64(this->numericUpDown3->Value), System::Decimal::ToDouble(this->numericUpDown1->Value), System::Decimal::ToDouble(this->numericUpDown2->Value), &coo);
		COO_To_MM(&coo, gen_filename.c_str());
		generateMatrixImage(&coo);
		FreeCOO(&coo);
		showCompletedMessage();
	}

	private: System::Void button3_Click(System::Object^ sender, System::EventArgs^ e)
	{
		FILE* f;
		struct coo_t coo;
		//
		msclr::interop::marshal_context context;
		std::string gen_filename = (INPUT_FOLDER + (std::string)"/" + GENERATOR_FOLDER + (std::string)"/" + context.marshal_as<std::string>(this->textBox2->Text) + ".mtx");
		if (createOutputDirectory(INPUT_FOLDER, GENERATOR_FOLDER))
		{
			showOutputDirErrorMessage();
			return;
		}
		//
		generateMatrixGaussMethodCol(System::Decimal::ToUInt64(this->numericUpDown3->Value), System::Decimal::ToDouble(this->numericUpDown4->Value), System::Decimal::ToDouble(this->numericUpDown5->Value), &coo);
		COO_To_MM(&coo, gen_filename.c_str());
		generateMatrixImage(&coo);
		FreeCOO(&coo);
		showCompletedMessage();
	}

	private: System::Void button4_Click(System::Object^ sender, System::EventArgs^ e)
	{
		FILE* f;
		struct coo_t coo;
		//
		msclr::interop::marshal_context context;
		std::string gen_filename = (INPUT_FOLDER + (std::string)"/" + GENERATOR_FOLDER + (std::string)"/" + context.marshal_as<std::string>(this->textBox2->Text) + ".mtx");
		if (createOutputDirectory(INPUT_FOLDER, GENERATOR_FOLDER))
		{
			showOutputDirErrorMessage();
			return;
		}
		//
		generateMatrixGaussMethodFull(System::Decimal::ToUInt64(this->numericUpDown3->Value), System::Decimal::ToDouble(this->numericUpDown1->Value), System::Decimal::ToDouble(this->numericUpDown2->Value), System::Decimal::ToDouble(this->numericUpDown4->Value), System::Decimal::ToDouble(this->numericUpDown5->Value), &coo);
		COO_To_MM(&coo, gen_filename.c_str());
		generateMatrixImage(&coo);
		FreeCOO(&coo);
		showCompletedMessage();
	}

	private: System::Void button7_Click(System::Object^ sender, System::EventArgs^ e)
	{
		skipMethodRowCheckInput();
		//
		FILE* f;
		struct coo_t coo;
		//
		msclr::interop::marshal_context context;
		std::string gen_filename = (INPUT_FOLDER + (std::string)"/" + GENERATOR_FOLDER + (std::string)"/" + context.marshal_as<std::string>(this->textBox2->Text) + ".mtx");
		if (createOutputDirectory(INPUT_FOLDER, GENERATOR_FOLDER))
		{
			showOutputDirErrorMessage();
			return;
		}
		//
		generateMatrixImbalancedRow(System::Decimal::ToUInt64(this->numericUpDown3->Value), System::Decimal::ToUInt64(this->numericUpDown6->Value), System::Decimal::ToUInt64(this->numericUpDown7->Value), &coo);
		COO_To_MM(&coo, gen_filename.c_str());
		generateMatrixImage(&coo);
		FreeCOO(&coo);
		showCompletedMessage();
	}

	private: System::Void button6_Click(System::Object^ sender, System::EventArgs^ e)
	{
		skipMethodColCheckInput();
		//
		FILE* f;
		struct coo_t coo;
		//
		msclr::interop::marshal_context context;
		std::string gen_filename = (INPUT_FOLDER + (std::string)"/" + GENERATOR_FOLDER + (std::string)"/" + context.marshal_as<std::string>(this->textBox2->Text) + ".mtx");
		if (createOutputDirectory(INPUT_FOLDER, GENERATOR_FOLDER))
		{
			showOutputDirErrorMessage();
			return;
		}
		//
		generateMatrixImbalancedCol(System::Decimal::ToUInt64(this->numericUpDown3->Value), System::Decimal::ToUInt64(this->numericUpDown8->Value), System::Decimal::ToUInt64(this->numericUpDown9->Value), &coo);
		COO_To_MM(&coo, gen_filename.c_str());
		generateMatrixImage(&coo);
		FreeCOO(&coo);
		showCompletedMessage();
	}

	private: System::Boolean skipMethodRowCheckInput()
	{
		if (this->numericUpDown3->Value <= this->numericUpDown6->Value)
		{
			showSkipMethodErrorMessage();
			return false;
		}
		return true;
	}

	private: System::Boolean skipMethodColCheckInput()
	{
		if (this->numericUpDown3->Value <= this->numericUpDown8->Value)
		{
			showSkipMethodErrorMessage();
			return false;
		}
		return true;
	}

	private: System::Void showSkipMethodErrorMessage()
	{
		System::Windows::Forms::MessageBox::Show(this, "Start cannot be equal to or higher than matrix dimension !", \
			"Error", \
			System::Windows::Forms::MessageBoxButtons::OK);
	}

	private: System::Void showCompletedMessage()
	{
		System::Windows::Forms::MessageBox::Show("Matrix has been created and saved !", \
			"Done !", \
			System::Windows::Forms::MessageBoxButtons::OK);
	}

	private: System::Void showOutputDirErrorMessage()
	{
		System::Windows::Forms::MessageBox::Show(this, "Output directory could not be created !", \
			"Error", \
			System::Windows::Forms::MessageBoxButtons::OK);
	}
};
}