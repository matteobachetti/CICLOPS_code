# Generated by Django 3.0.2 on 2020-02-28 09:38

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('polls2', '0006_auto_20200226_1051'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='pulseprofile',
            name='disp_profile1',
        ),
    ]