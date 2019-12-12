#include "SpMVM_GEN_MAT_FORM.h"

using namespace System;
using namespace System::Windows::Forms;

[STAThreadAttribute]
void main(array<String^>^ args) {
    Application::EnableVisualStyles();
    Application::SetCompatibleTextRenderingDefault(false);
    //WinformCDemo is your project name
    SpMVMGENMAT::SpMVM_GEN_MAT_FORM form;
    Application::Run(% form);
}